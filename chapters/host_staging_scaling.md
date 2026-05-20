# Host-Staged GPU Collectives in oneCCL: Mechanisms, Models, and Scaling Implications

## Abstract

Intel Xe GPU architectures (PVC, BMG/CRI) require all inter-node collective
communication to pass through host DRAM by default. This "host bounce buffer"
design causes two distinct scaling failure modes:

1. **Latency accumulation (small messages, 8–16 nodes).** For decode-time
   tensor parallelism, where messages are small and latency-critical, the
   per-step host-staging overhead compounds across collective algorithm steps.
   The alpha-beta model in §5 estimates that this overhead exceeds per-layer
   compute time in the 8–16 node range for batch-1 decode.

2. **Throughput/congestion saturation (large messages, 1000+ nodes).** For
   training gradient allreduce and MoE alltoallv at scale, the symptom is
   different: simultaneous D2H and H2D staging from many tiles saturates local
   PCIe bandwidth and host DRAM throughput, capping effective collective
   bandwidth below what the network fabric can deliver (§6–7).

Both modes share the same root cause — data transits host memory on every
inter-node step — but manifest at different scales and message sizes. The
bounce buffer cost comes from the software data path, not from network
bandwidth, so it cannot be removed by algorithm selection alone without
changing how data reaches the NIC. This chapter traces the mechanism through
oneCCL source code, derives the scaling behavior for both regimes, and
identifies deployment conditions under which each becomes dominant.

**Reader map:**
- §1-2: Mechanism — the D2H→allreduce→H2D chain, HMEM bypass, per-step latency model
- §3-5: Decode scaling — alpha-beta derivation, bottlenecks, 8-16 node crossover for TP
- §6: Training at scale — gradient bandwidth saturation, ZeRO, PCIe throughput limits
- §7: MoE at scale — alltoallv congestion from simultaneous staging across tiles
- §8-9: Evidence — Aurora measurements, GPUDirect RDMA contrast
- §10: Deployment guidance — runtime configuration, HMEM activation, mitigation limits

**If you are debugging large-scale training or MoE (100+ nodes),** start at §6-7.
The decode model in §3-5 is for small-message TP; the mechanism in §2 applies to both.

---

## 1. Collective Communication Requirements

oneCCL is Intel's collective communication library. It implements allreduce,
broadcast, all-to-all, and other operations that are on the critical path of
distributed training and inference. The host-staging mechanism described in §2
applies to all inter-node collectives regardless of message size or node count.
Its scaling consequences differ by workload regime:

- **Tensor parallelism (8–64 tiles, small messages):** Each TP allreduce is
  latency-critical. Host staging adds a fixed per-step cost that dominates the
  message transfer time. §3-5 model this regime.
- **Data parallelism / ZeRO (100–10,000+ nodes, large messages):** Gradient
  allreduce is bandwidth-critical. Host staging forces all data through the
  node's PCIe/host DRAM path, capping effective throughput below network
  capacity. §6 models this regime.
- **MoE expert parallelism (10–1000+ nodes, medium messages, all-to-all):**
  Every tile simultaneously stages data for every other tile. The N×N traffic
  pattern concentrates all staging on the same PCIe root complexes. §7 models
  this regime.

The common mechanism is a sequential D2H→collective→H2D chain that every
inter-node message traverses by default. For small messages, the symptom is
per-step latency accumulation. For large messages at scale, the symptom is
PCIe and host DRAM bandwidth saturation. Both map to the same code path in
oneCCL (§2).

---

## 2. Host-Staged (Bounce Buffer) Scaleout Mechanism

On PVC and BMG/CRI, GPU kernels cannot issue network operations. The NIC is not
accessible from GPU-side code. Data must bounce through host DRAM on both
the send and receive sides:

```
Default path: OFI transport, no HMEM

  Node A                                               Node B
  ┌───────────┐                                       ┌───────────┐
  │  GPU VRAM │                                       │  GPU VRAM │
  └─────┬─────┘                                       └─────▲─────┘
   D2H  │ PCIe                                         H2D  │ PCIe
  ~2 us ▼                                             ~2 us │
  ┌───────────┐                                       ┌─────┴─────┐
  │ Host DRAM │                                       │ Host DRAM │
  │ (staging) │                                       │ (staging) │
  └─────┬─────┘                                       └─────▲─────┘
   CPU  │ fi_tsendmsg                             CPU wait  │
  ~1 us ▼                                             ~1 us │
  ┌───────────┐                                       ┌─────┴─────┐
  │   NIC A   │───────────── wire ~2 us ────────────▶ │   NIC B   │
  └───────────┘  NIC reads host DRAM                  └───────────┘
                                              NIC writes host DRAM

  All 8 stages are sequential. Total: ~8-10 us per collective step.
  Network-only cost would be ~2-3 us.


Experimental path: OFI + CCL_ATL_HMEM=1

  Node A                                               Node B
  ┌───────────┐                                       ┌───────────┐
  │  GPU VRAM │                                       │  GPU VRAM │
  └─────┬─────┘                                       └─────▲─────┘
  PCIe  │ (NIC DMA-reads GPU BAR)          (NIC DMA-writes) │ PCIe
  ~1 us ▼                                             ~1 us │
  ┌───────────┐                                       ┌─────┴─────┐
  │   NIC A   │───────────── wire ~2 us ────────────▶ │   NIC B   │
  └───────────┘                                       └───────────┘
  CPU still calls fi_tsendmsg, but data never touches host DRAM.
  Total: ~4 us per step.
```

On the non-HMEM path, all inter-node communication follows this sequence
(from `allreduce_scaleout_sycl.cpp`, lines 29-100):

```
Send side:
  (1) GPU kernel writes output to GPU VRAM
  (2) SYCL memcpy: GPU VRAM → pinned host buffer      [D2H over PCIe]
      q.submit([=](handler& h) { h.memcpy(host_buf, gpu_buf, size); })
  (3) host_task: atl_comm->allreduce(host_buf)         [CPU posts to NIC]
  (4) NIC DMA-reads host DRAM → wire                   [PCIe + network]

Receive side:
  (5) NIC DMA-writes received data → host buffer       [PCIe]
  (6) atl_comm->check() → if not done, atl_comm->wait() [CPU blocks]
  (7) SYCL memcpy: host buffer → GPU VRAM              [H2D over PCIe]
      q.submit([=](handler& h) { h.memcpy(recv_buf, host_buf, size); })
  (8) GPU kernel reads result from VRAM
```

These steps are **sequential with no overlap**. The code at
`allreduce_scaleout_sycl.cpp` lines 62-96 makes this explicit: the D2H
`memcpy` event must complete before `host_task` is submitted, and the
`host_task` (`atl_comm->allreduce` + `wait`) must complete before the H2D
`memcpy` is submitted. There is a `TODO: chunking/pipelining` comment at
line 33 acknowledging this sequentiality is unresolved.

The OFI path forces this sequence regardless of other settings.
`allreduce_scaleout_sycl.cpp` lines 116-121:

```cpp
bool copy_to_host = ccl::global_data::env().sycl_enable_direct_gpu_rdma ? false : true;
ze_device_handle_t ze_dev = ...;
if (should_disable_rdma(ze_dev) || ccl::global_data::env().atl_transport == ccl_atl_ofi) {
    copy_to_host = true;
}
```

Even if `CCL_SYCL_ENABLE_DIRECT_GPU_RDMA=1` is set, line 119 overrides it to
`true` whenever `atl_transport == ccl_atl_ofi`. In the default OFI
configuration (without `CCL_ATL_HMEM=1`), any deployment without a confirmed
HMEM path operates in the host-staged regime. The HMEM path (§2.0) also uses OFI but
bypasses host DRAM by having the NIC DMA from the GPU BAR directly; it
requires hardware capability probing at communicator init and is currently
experimental (see §2.0 for requirements and failure modes).

The `topo` algorithm's scaleout phase has the same gate in `coll_util.cpp`:

```cpp
bool enable_hmem = (ccl::global_data::env().use_hmem && atl_base_comm::attr.out.enable_hmem);
if (!enable_hmem) {
    LOG_DEBUG("topo/scale_out: use host_", ccl_coll_type_to_str(coll_param.ctype));
    // allocates host staging buffers, performs D2H, runs collective, H2D
}
```

Both code paths land in the same place: sequential D2H, network collective,
H2D — no overlap between stages.

### 2.1 HMEM Bypass Path

The diagram above shows "OFI + CCL_ATL_HMEM=1" as a path where data never
touches host DRAM. This seems to contradict the `copy_to_host = true` override
at line 119. They are different code paths: **HMEM and `allreduce_scaleout_sycl_simple` share no code.**

The `coll_util.cpp` gate is the branching point:

```cpp
bool enable_hmem = (ccl::global_data::env().use_hmem && atl_base_comm::attr.out.enable_hmem);
if (!enable_hmem) {
    // → allreduce_scaleout_sycl_simple: SYCL D2H memcpy + host_task + H2D memcpy
} else {
    // → atl_ofi.cpp FI_HMEM path: NIC DMA-reads GPU BAR directly, memcpy chain never runs
}
```

When `CCL_ATL_HMEM=1` is set and the hardware supports it, execution takes the
`enable_hmem == true` branch:

1. During communicator init, `atl_ofi.cpp` probes the libfabric provider for
   `FI_HMEM` capability and registers GPU memory using Linux dmabuf. This
   tells the NIC: "this address range is GPU BAR memory — DMA from it directly."
2. When `fi_tsendmsg()` is called, the NIC DMA-reads from the GPU BAR over
   PCIe without any CPU-side SYCL memcpy. The `allreduce_scaleout_sycl_simple`
   function and its `copy_to_host = true` override are never reached.
3. The CPU still calls `fi_tsendmsg()` (the GPU cannot post network operations
   autonomously), but data goes GPU BAR → NIC → wire → NIC → GPU BAR via
   PCIe, skipping host DRAM entirely.

The `copy_to_host = true` line in `allreduce_scaleout_sycl.cpp` only overrides
the `CCL_SYCL_ENABLE_DIRECT_GPU_RDMA` knob within the sycl scaleout path. That
code is only entered when `enable_hmem == false`. There is no contradiction.

**Requirements for the HMEM path:** `CCL_ATL_HMEM=1` requires a libfabric provider with `FI_HMEM_ZE`
support (Intel Level Zero GPU memory). oneCCL's memory registration identifies
Intel GPU allocations via `zeMemGetAllocProperties` and registers them with
`fi_mr_regattr(iface=FI_HMEM_ZE)`. Additional OS/driver requirements:

- Linux kernel ≥ 5.12 with `CONFIG_DMA_BUF` and Intel GPU driver dmabuf enabled
- Intel GPU driver (i915/xe) with P2P dmabuf support active

**The hardware and driver stack on Aurora are capable.** The PVC xe driver
exports GPU BAR memory as Linux dmabuf via Level Zero's
`ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF`. The Cassini NIC can DMA from
dmabuf-registered memory (Allcock et al. confirm GPU Direct RDMA via dma-buf
P2P DMA works through MPICH on Aurora, demonstrating that the platform
supports GPU-memory DMA through this mechanism). libfabric's util-layer
`src/hmem_ze.c` is a complete implementation — behind `#if HAVE_ZE` — that the
CXI provider routes through via the shared `hmem_ops[FI_HMEM_ZE]` dispatch.

**Whether `FI_HMEM_ZE` is working in the libfabric deployed on Aurora is what
determines whether the HMEM path is usable.** The most likely gate is the `HAVE_ZE` compile flag, but
driver version, kernel dmabuf support, and NIC firmware can all affect whether
the probe succeeds. If it fails, oneCCL falls back to host staging silently.
The CXI provider also exposes a `force_ze_hmem_support` environment variable,
indicating ZE was explicitly anticipated but treated as off-by-default.

Always verify with `CCL_LOG_LEVEL=info`: if `"use_hmem: 1"` does not appear in
startup output, HMEM fell back to staging regardless of the flag setting.

### 2.2 Staging Buffer Limits

The host staging buffer is pre-allocated at communicator init time. When the
message size exceeds this pre-allocated `scaleout_host_buf_size`, the direct SYCL path
sets `done = false` and falls back to a different schedule (the outer loop).

While this fallback mechanism is intended to handle chunks iteratively, relying
on the outer loop rather than a tight, overlapped transmission pipeline (as seen
in NCCL) contributes to inefficiencies for giant payloads like massive gradient
syncs or full-KV allgathers.

### 2.3 Message-Size-Based Algorithm Selection

`selector.hpp` defines the size thresholds:

```cpp
#define CCL_ALLREDUCE_SHORT_MSG_SIZE   8192        // 8192 elements (not bytes)
#define CCL_ALLREDUCE_MEDIUM_MSG_SIZE  (1024*1024) // 1M elements
```

The thresholds are in **element count**. For BF16 (2 bytes/element), SHORT is < 16 KB
and MEDIUM is < 2 MB. The `// 8 KB` comment in the source is only accurate for 1-byte
element types.

For the SYCL+ZE build (GPU path, `selector_allreduce.cpp` lines 20-46):
- Main algorithm: `topo` for all message sizes
- Scaleout (SYCL path): `direct` for BF16 ≤ 1-4 MB depending on comm_size
  (delegates to the underlying MPI `MPI_Allreduce`; the sub-linear latency
  scaling in Ibeid et al. Figure 14 is consistent with recursive doubling, but
  the MPI algorithm selection is not established from oneCCL source). Ring or
  rabenseifner for larger messages. Selection logic: `sycl_selection.cpp:330-380`.
- Scaleout (scheduler fallback path): `ring` for all sizes (`scaleout_table`, line 46)
- Fallback: `recursive_doubling` for 0–8192 elements (< 16 KB BF16), `ring` above

For the CPU/non-ZE build with OFI:
- 0–8192 elements (< 16 KB BF16): `recursive_doubling`
- 8193–1,048,576 elements (16 KB – 2 MB BF16): `nreduce` (reduce-scatter + allgather)
- > 1,048,576 elements (> 2 MB BF16): `ring`

The ring algorithm has a limited overlap window
(`is_copy_overlap_enabled()` in `coll_util.cpp`): overlap only activates for
ring + multi-node + single-worker-mode. For the default scaleout path through
`allreduce_scaleout_sycl_simple`, there is no overlap at all.

### 2.4 Per-Step Latency Model

For a 16 KB message on a PVC/BMG/CRI node (PCIe Gen4/5, ~32 GB/s effective):

| Operation | Notes | Latency |
|---|---|---|
| D2H SYCL memcpy (GPU → host buf) | PCIe, contended with peer tiles | ~1-3 us |
| host_task submission overhead | SYCL host task scheduling | ~0.5 us |
| OFI post (fi_tsendmsg) | CPU software, no GPU involvement | ~1-2 us |
| NIC DMA (host DRAM → wire) | 25 GB/s effective | ~0.5 us |
| Network transit (Slingshot-11) | hardware latency | ~1-2 us |
| NIC DMA (wire → host DRAM) | 25 GB/s effective | ~0.5 us |
| atl_comm->wait() spin | CPU polling for completion | ~0.5 us |
| H2D SYCL memcpy (host buf → GPU) | PCIe, contended | ~1-3 us |

**Total per step with bounce buffer:** ~6-12 us  
**Network-only latency per step:** ~2-4 us  
**Overhead ratio:** 2-4x

---

## 3. Collective Algorithm Scaling Under Host Staging

### 3.1 Recursive Doubling

Recursive doubling (`allreduce.cpp` lines 525-639) executes log₂(N) steps.
Each step exchanges M bytes bidirectionally. Total latency:

```
T_rd = log₂(N) * (2*α + M/β)
```

where α is **one-way** per-hop latency and β is bandwidth. (The standard
Thakur/Gropp notation folds both directions into one α; our formulation is
equivalent — each step incurs a send-side staging cost α and a receive-side
staging cost α, hence 2α per step.) Without bounce buffer, α is the hardware
network latency (~2 us on Slingshot-11). With bounce buffer, α becomes the
sum of all staging stages on one side:

```
α_staged = α_D2H + α_host_task + α_OFI + α_network + α_wait + α_H2D
         ≈ 2 + 0.5 + 1.5 + 1.5 + 0.5 + 2 = 8 us
```

For a 16 KB BF16 message (8192 elements — exactly at the SHORT/MEDIUM boundary,
oneCCL's SYCL+ZE path selects `topo` → `direct` scaleout → MPI `MPI_Allreduce`.
The sub-linear (log₂N) latency scaling in Ibeid et al., Figure 14 is consistent
with recursive doubling; the model below uses log₂(N) steps on that basis.
The same per-step structure applies to any algorithm with logarithmic step count):


| N nodes | log₂(N) steps | T without staging | T with staging |
|---|---|---|---|
| 8 | 3 | 3 * (4 + 0.5) = 14 us | 3 * (16 + 0.5) = 50 us |
| 64 | 6 | 6 * (4 + 0.5) = 27 us | 6 * (16 + 0.5) = 99 us |
| 512 | 9 | 9 * (4 + 0.5) = 41 us | 9 * (16 + 0.5) = 149 us |
| 2048 | 11 | 11 * (4 + 0.5) = 50 us | 11 * (16 + 0.5) = 182 us |

The Aurora benchmark (Ibeid et al., arXiv:2512.04291, Figure 14:
"Latency for MPI reduction operation for buffers located in GPU memory")
measures ~250 us at 2048 nodes for 8-byte messages. The gap between the
182 us model above and 250 us accounts
for: oneCCL scheduler overhead, SYCL event graph management, atl_comm
endpoint contention, and the intra-node Xe Link phases within each PVC node
that are serialized before the inter-node step.

The model estimates **~130 us of added allreduce latency at 2048 nodes compared
to a hypothetical direct GPU RDMA path** — the gap between the staged α and the
hardware network latency, accumulated across 11 steps.

### 3.2 Ring Allreduce

Ring allreduce executes 2(N-1) steps, each moving M/N bytes. With N large,
this collapses to:

```
T_ring ≈ 2*N * α + 2*M/β
```

```
Ring allreduce (4 GPUs, M bytes, broken into 4 chunks A, B, C, D):

  Step 1 (reduce-scatter): Every GPU sends ONE chunk (M/4 bytes).
    GPU0 sends A_0 → GPU1        GPU1 sends B_1 → GPU2
    GPU2 sends C_2 → GPU3        GPU3 sends D_3 → GPU0

    GPU1 accumulates: A_0+A_1    GPU2: B_1+B_2    GPU3: C_2+C_3    GPU0: D_3+D_0

  Steps 2-3 (reduce-scatter): GPUs forward the newly accumulated partial sums.
    After 3 steps, each GPU holds exactly one fully-reduced chunk:
    GPU0 holds sum(B)   GPU1 holds sum(C)   GPU2 holds sum(D)   GPU3 holds sum(A)

  Steps 4-6 (allgather): Distribute the completed chunks.
    GPU0 passes sum(B) → GPU1.  GPU1 passes sum(C) → GPU2.  etc.
    After 3 steps, all GPUs have the full sum(A, B, C, D).

  Result: 2(N-1) = 6 steps.  Per-GPU traffic: 2(N-1)/N × M = 1.5M.
  No redundant data moved — optimal bandwidth utilization for large M.
```

Ring's O(N) latency scaling makes it unsuitable for large-N inference.
oneCCL uses ring only for the scaleout phase at message sizes > 8192 elements (> 16 KB BF16) where
bandwidth efficiency matters more than latency. For decode inference (small
messages, latency critical), recursive doubling is selected automatically
via the threshold in `selector_allreduce.cpp`.

### 3.3 Hierarchical `topo` Allreduce

`topo` reduces inter-node traffic by performing an intra-node reduce-scatter
first (Xe Link on PVC, PCIe P2P on BMG/CRI), then running the scaleout collective
on 1/T of the data, where T is the number of tiles per node. The inter-node
phase still goes through host staging. For a PVC node with 12 tiles:

```
T_topo = T_xe_link_reduce_scatter + T_scaleout(M/12)
       = ~10 us + log₂(N_nodes) * (16 + M/(12*β)) us
```

For small M (decode, 16 KB), the M/12 bandwidth term is negligible and the
latency floor is:

```
T_topo ≈ 10 + log₂(N) * 16 us
```

At 2048 nodes: `T_topo ≈ 10 + 11*16 = 186 us`

This is consistent with the Aurora measurements, but the microbenchmark does
not by itself validate the decode crossover — it measures collective latency at
scale, not end-to-end decode efficiency. The Xe Link phase on PVC amortizes
some PCIe contention (fewer tiles staging per node), but does not change the
per-step inter-node latency.

---

## 4. Sources of Scaling Overhead

### 4.1 Host DRAM Bandwidth Pressure

On a PVC node with 12 tiles, all 12 tiles stage through host DRAM
simultaneously during the reduce-scatter phase. The staging path is:

```
GPU VRAM → PCIe → CPU DRAM → PCIe → NIC → wire
```

The bandwidth ceiling at each hop:

| Path | Aurora (PVC) | Typical BMG/CRI |
|---|---|---|
| PCIe GPU→CPU | Gen5 x16 = 64 GB/s per GPU, 6 GPUs → 384 GB/s aggregate | Gen5 x16, fewer GPUs |
| CPU DRAM | DDR5-4800 8-channel Sapphire Rapids ≈ 307 GB/s per socket | Lower |
| PCIe CPU→NIC | Separate lanes from GPU PCIe | Separate |
| NIC aggregate | 8 × 200 Gbps = 200 GB/s (Ibeid et al., Fig. 1) | 2–4 × 100 Gbps |

Aurora's GPU→PCIe bandwidth (384 GB/s) exceeds NIC demand (200 GB/s), so PCIe
itself is not the bottleneck. The actual constraint when all 12 tiles stage
simultaneously is **CPU DRAM bandwidth**: 12 tiles writing 16 KB each in one
step = 192 KB, but under continuous decode at high token rate, the aggregate
D2H + H2D demand can approach the ~307 GB/s DDR5 ceiling, leaving the NICs
underutilized.

For individual message latency (the dominant bottleneck for decode), the D2H
transfer time is:

```
T_D2H = 16 KB / 64 GB/s (uncontended, single GPU) ≈ 0.25 us
T_D2H = 16 KB / (64 GB/s / 2 tiles per GPU) ≈ 0.5 us  (2 tiles sharing one GPU link)
```

Contention with concurrent staging from other ranks on the same CPU socket
pushes this toward ~1-3 us in practice — consistent with the §2.3 budget table.

This explains why Aurora measures ~23 GB/s effective per NIC (Ibeid et al.,
Figure 12) rather than the 25 GB/s theoretical maximum: under concurrent
staging, host DRAM bandwidth becomes the shared resource and the CPU-side
posting overhead limits NIC utilization, not PCIe bandwidth to the GPU.

The PCIe Gen4 NIC path (32 GB/s per NIC) is confirmed by two independent
sources: Ibeid et al. (arXiv:2512.04291) cite "PCIe Gen4-PCIe Gen5 conversion
inefficiencies" explicitly as the cause of the ~23 GB/s measured bandwidth for
GPU-memory buffers, and Goto et al. (arXiv:2604.09517) describe the Aurora
fabric geometry as "PCIe switches that fan out Gen5 x16 lanes to Gen4 x16
endpoints" for the NIC-facing ports.

### 4.2 Sequential Copy and Network Stages

The `allreduce_scaleout_sycl_simple` path (lines 29-100) chains three
sequential SYCL submissions via event dependencies:

```
[D2H memcpy] → [host_task(allreduce+wait)] → [H2D memcpy]
```

Each submission depends on the previous completing. The `TODO:
chunkingThe SYCL Host Task Scheduler Bottleneck

The actual critical software gap in the default scale-out path is how oneCCL
orchestrates the CPU's involvement using **SYCL Host Tasks**.

When oneCCL issues network communication, it queues it as a SYCL `host_task`.
The code chains three sequential SYCL submissions via event dependencies:

```
[D2H memcpy] → [host_task(allreduce+wait)] → [H2D memcpy]
```

This design subjects the scale-out communication to the constraints of the
underlying SYCL runtime:

1. **Scheduling Overhead:** SYCL's host task scheduler tracks task readiness
   via an internal dependency queue. The SYCL runtime searching the queue for
   the next task to execute becomes extremely inefficient (a linear search) as
   the sequence of nested dependencies grows.
2. **Blocking GPU Tasks:** Because these host tasks have event dependencies with
   Level Zero (GPU) tasks, a slow host task executing on the CPU (waiting for
   long-tail network completions via `atl_comm->wait()`) effectively blocks
   subsequent Level Zero execution. The queue rapidly builds up.
3. **No Overlap Constraint:** oneCCL fundamentally does not support processing
   overlapping collective calls within its core scheduling engine in this manner.

According to Intel oneCCL developers, the accumulation of this queue overhead
is the primary driver of poor scaling on the host-staged path, prompting efforts
to bypass `host_task` completely and invoke direct algorithm CPU-GPU memcpys
and raw MPI paths for future releases.

### 4.3 Worker Thread Queu
```
collectives/second = tokens/second * allreduces_per_token
                   = tokens/second * 160   (80-layer model, 2 allreduces/layer)
```

At 100 tokens/second, that is 16,000 collectives/second per rank. Each takes
~200 us at 2048 nodes. The steady-state queue depth at the worker is:

```
queue_depth = arrival_rate * service_time = (16000/s) * (0.0002 s) = 3.2
```

A queue depth above 1 means the worker is saturated and collectives wait
before being issued. This queuing delay adds to the per-collective latency
monotonically with node count (higher node count → longer service time →
higher queue depth → longer queuing delay). This is a second-order effect on
top of the staging overhead.

To observe this saturation in real-time, compile oneCCL with ITT enabled
(`CCL_ENABLE_ITT=1`) and use Intel VTune Profiler; the
`allreduce_scaleout_sycl_simple` ITT markers will show significant gaps
between task submission and execution when the worker reaches queue depth > 1.

---

## 5. Decode-Time Allreduce Scaling Model

The scaling wall is the node count N* where `T_communication >= T_compute`
per layer. For decode inference:

```
T_compute per layer ≈ weight_bytes_per_layer / (HBM_BW * N_tiles_per_node * N_nodes)
T_comm per allreduce ≈ T_intranode + log₂(N_nodes) * α_staged
```

(The log₂(N) term reflects a recursive-doubling step count, which is consistent
with the sub-linear scaling in Ibeid et al. Figure 14. The SYCL topo scaleout
path uses `direct` and delegates to MPI's native allreduce — see §2.2 and
`sycl_selection.cpp`. The MPI-internal algorithm selection is not visible from
oneCCL source.)

At batch=1 decode, GEMMs have shape [1,K]×[K,N] — arithmetic intensity ≈ 1
FLOP/byte, so execution is memory-bandwidth-limited, not compute-limited.
The tile spends its time streaming weight matrices from HBM, not waiting for
FMAs to finish. T_compute is therefore determined by HBM bandwidth, not peak
TFLOPS.

For a representative 70B model on PVC-class hardware:
- HBM bandwidth per tile: ~100 GB/s **effective for batch=1 matrix-vector GEMMs**
  (PVC Max 1550 peak HBM is 1.638 TB/s per tile, but batch=1 [1,K]×[K,N] GEMMs
  achieve roughly 5–10% of peak HBM due to small working sets, cache-line
  utilization, and memory controller overhead — giving ~80–160 GB/s sustained.
  100 GB/s is a conservative estimate; the wall crossover shifts by ±4 nodes
  if the true effective bandwidth is 200 GB/s. BMG/CRI: similar effective BW
  per tile at batch=1.)
- Weight bytes per layer (BF16): ~1.6 GB (SwiGLU MLP: 3 matrices + attention
  projections with GQA; see concrete Llama-3 70B example below for derivation)
- T_compute per layer (1 tile, batch=1): ~1640 MB / 100 GB/s ≈ 16 ms (entire
  layer, unsharded)
- With TP across T tiles: T_compute = 16 ms / T
- T_intranode (Xe Link): ~10 us
- α_staged (bounce buffer): ~8-12 us per hop

The wall condition is: `T_comm >= T_compute_per_tile`:

```
T_intranode + log₂(N_nodes) * α_staged >= weight_bytes / (HBM_BW * T_total)
```

where `T_total = tiles_per_node × N_nodes`. As N increases, T_compute
shrinks (more tiles sharing the layer) while T_comm grows (more hops).

For PVC with 12 tiles/node, TP across all tiles, at N=8 nodes (96 tiles):
```
T_compute = 1640 MB / (100 GB/s × 96) ≈ 171 µs
T_comm = 10 + log₂(8) × 10 = 40 µs (per allreduce)
Total comm per layer (2 allreduces) = 80 µs
```

Communication fraction = 80 / (171 + 80) ≈ 32%. Significant but not dominant.
However, this is per-allreduce latency that cannot overlap with compute (the
allreduce result feeds the next layer's GEMM input), so it adds directly to
time-per-token.

At N=16 nodes (192 tiles):
```
T_compute = 1640 MB / (100 GB/s × 192) ≈ 85 µs
T_comm = 10 + log₂(16) × 10 = 50 µs (per allreduce)
Total comm per layer = 100 µs
Communication fraction = 100 / (85 + 100) ≈ 54%
```

At N=16, communication exceeds compute per layer. Adding more nodes
reduces T_compute further but T_comm continues growing — diminishing returns
set in sharply.

Under these assumptions, N* ≈ 8-16 nodes. Beyond this, allreduce dominates the
per-token budget in the model.

### 5.1 Representative Llama-3 70B Decode Case

For a specific, measurable case: Llama-3 70B, tensor-parallel across N nodes
with 8 tiles per node (one rank per tile), batch=1 decode, BF16.

**Model parameters:**
- d_model = 8192, d_ffn = 28672, 80 layers
- Llama-3 uses SwiGLU: three MLP weight matrices (gate_proj, up_proj, down_proj)
  plus four attention projections (Q, K, V, O)
- MLP weight bytes per layer (BF16): 3 × d_model × d_ffn × 2 = 3 × 8192 × 28672 × 2 ≈ 1.34 GB
- Attention weight bytes per layer (BF16): Q [8192,8192] + K [8192,1024] +
  V [8192,1024] + O [8192,8192] (GQA with 8 KV heads) × 2 ≈ 0.30 GB
- Total weight bytes per layer: ~1.64 GB (dominates batch=1 execution time)
- Allreduce message is the **output activation**, not the weight matrix:
  allreduce message = batch × d_model × 2 bytes = 1 × 8192 × 2 = 16 KB

**Compute per tile at batch=1 (memory-bandwidth-limited):**

At batch=1, GEMMs have shape [1,K]×[K,N]. Arithmetic intensity ≈ 1 FLOP/byte.
The tile streams weight matrices from HBM; execution time is determined by
memory bandwidth, not TFLOPS.

- HBM BW per tile: ~100 GB/s
- Weight bytes per layer sharded over TP=8 tiles per node: 1.64 GB / 8 = 205 MB
- T_compute per tile per layer: 205 MB / 100 GB/s ≈ **2.0 ms** (unsharded
  across nodes — one node handles the full layer, 8 tiles share it)
- With N nodes (TP across all N×8 tiles): T_compute = 1640 MB / (100 GB/s × 8 × N)

**Communication at N=8 nodes:**

The topo algorithm performs intra-node reduce-scatter first (T_intranode),
then runs the inter-node scaleout across N_nodes representatives (one per node).
Inter-node steps = log₂(N_nodes) = log₂(8) = 3.

```
T_comm = T_intranode + log₂(N_nodes) * α_staged
       = 10 µs + 3 * 10 µs
       = 40 µs
```

T_compute at N=8 nodes (64 tiles total):
```
T_compute = 1640 MB / (100 GB/s × 64) ≈ 256 µs per layer
```

With 2 allreduces per layer (attention + MLP): total comm = 2 × 40 = 80 µs.
Communication fraction = 80 / (256 + 80) ≈ **24%** — significant overhead,
but not yet dominant.

**At N=16 nodes (128 tiles):**
```
T_compute = 1640 MB / (100 GB/s × 128) ≈ 128 µs
T_comm = 10 + log₂(16) × 10 = 50 µs (per allreduce)
Total comm per layer = 2 × 50 = 100 µs
Communication fraction = 100 / (128 + 100) ≈ 44%
```

**At N=64 nodes (512 tiles):**
```
T_compute = 1640 MB / (100 GB/s × 512) ≈ 32 µs
T_comm = 10 + log₂(64) × 10 = 70 µs (per allreduce)
Total comm per layer = 2 × 70 = 140 µs
Communication fraction = 140 / (32 + 140) ≈ 81% — wall-limited
```

The pure-GEMM wall (T_comm > T_compute_GEMM) is crossed between 16 and 64 nodes.
The abstract's "8–16 node" estimate incorporates two additional effects:

**Amdahl serial fraction.** At batch=1, non-GEMM operations (RoPE, RMSNorm,
KV cache read/write, softmax) account for roughly 20–30% of per-layer wall
time and do not parallelize across TP. Amdahl's law gives the maximum speedup:

```
Speedup(N) = 1 / (f_serial + (1 - f_serial) / N)

With f_serial = 0.25 (25% non-parallelizable):
  N=16:  max speedup = 1 / (0.25 + 0.75/16) = 3.4×  (vs theoretical 16×)
  N=64:  max speedup = 1 / (0.25 + 0.75/64) = 3.8×  (almost no gain over N=16)
```

The practical efficiency floor from Amdahl alone limits useful TP to ~8–16 nodes
(96–192 tiles at 12 tiles/node) before the serial fraction dominates —
independent of communication overhead. When communication overhead is added on
top, the effective wall appears earlier.

**Limited async overlap.** `async_op=True` can hide at most the non-GEMM work
between layers (~5–15 µs). At N=8 nodes T_comm ≈ 40 µs, so only ~25–35% of
the communication cost is hideable.

Combining both effects: for practical Llama-3 70B decode on PVC, adding nodes
beyond 8–16 yields diminishing returns even before the pure-GEMM wall is
reached at 16–64 nodes.

:::{note}
For MoE models (DeepSeek-R1/V3, Mixtral), the analogous scaling wall is
**alltoall-dominated**, not allreduce-dominated — see §5c below for the
full analysis and contrast with the Llama-3 allreduce wall.
:::

**Without the bounce buffer** (hypothetical direct GPU RDMA, α_network = 3 us):

At N=64 nodes with GPU RDMA:
```
T_comm = 10 + log₂(64) × 3 = 28 µs (per allreduce)
Total comm per layer = 2 × 28 = 56 µs
T_compute = 32 µs
```

Still wall-limited at 64 nodes, but compare the fractions: with host staging,
64 nodes gives 81% communication overhead. With GPU RDMA, the fraction would
be 56 / (32 + 56) ≈ 64%. The wall shifts from N≈16 to N≈64-128 — approximately
4× more headroom before hitting the same efficiency degradation.

The bounce buffer does not change the O(log N) scaling exponent of recursive
doubling. It multiplies the per-step coefficient by 3-4x, shifting the
scaling wall earlier by the same factor.

---

## 6. Training-Time Bandwidth Model

> **For Aurora at 1000+ nodes:** this section models a single-node BMG/CRI
> baseline. For Aurora-specific multi-rail throughput modeling, NIC utilization
> analysis, and tuning at 1000-10,000 nodes, see
> [Aurora at Scale: Training and MoE](aurora_large_scale).

The inference scaling analysis above addresses the **latency wall** — when allreduce latency
exceeds per-layer compute time for decode. Training has a different scaling failure mode:
the **bandwidth wall** — when gradient traffic exceeds the available PCIe/NIC bandwidth.

For inference, the question is: *is allreduce fast enough to stay off the critical path?*
For training, the question is: *is there enough bandwidth to move all gradients within
the step time?*

### 6.1 Gradient Volume at Scale

For a 7B model with BF16 parameters, the total gradient volume per step is:

```
7 × 10⁹ parameters × 2 bytes/param = 14 GB of gradients per step (DDP/ZeRO-0)
```

With ring allreduce, the data moved per rank is `2 × (p-1)/p × 14 GB`. At p=4:

```
Data per rank = 2 × 0.75 × 14 GB = 21 GB
```

At 25 GB/s effective PCIe P2P bandwidth (same single-node P2P path used for inference):

```
Time to move gradients = 21 GB / 25 GB/s = 840 ms
```

A 7B model forward + backward pass takes roughly 500–800 ms at batch=32 on a BMG/CRI node.
This means gradient communication is **already bandwidth-bound at 4 GPUs** with DDP.

### 6.2 Effect of ZeRO on Communication Volume

ZeRO-2 and ZeRO-3 do not reduce the *total data moved* — they change *when* the data
moves and how much is buffered at once:

```
DDP (ZeRO-0):
  Backward: one allreduce per layer, streamed
  Peak buffer: full model gradients (14 GB for 7B)
  Total data moved: 2 × (p-1)/p × 14 GB per step

ZeRO-2 (ReduceScatter instead of Allreduce):
  Backward: reduce-scatter per layer — each rank receives only its shard
  Peak buffer: 1/p of gradients (3.5 GB for 7B, p=4)
  Total data moved: same as DDP — ReduceScatter moves the same bytes as Allreduce
                    (both move 2 × (p-1)/p × n; allgather is skipped since each
                    rank only needs its own shard for the optimizer update)

ZeRO-3 (per-layer Allgather + ReduceScatter):
  Forward: allgather per layer to reconstruct params, discard after
  Backward: reduce-scatter per layer
  Peak buffer: 1/p of params (3.5 GB for 7B, p=4)
  Total data moved: 3 × (p-1)/p × 14 GB — MORE than DDP (extra allgather pass)
  Trade: memory shrinks from 14 GB to 3.5 GB; bandwidth cost increases 50%
```

ZeRO-3 trades bandwidth for memory. If you are already bandwidth-bound, ZeRO-3
makes it worse — only use it when per-GPU memory is the constraint.

### 6.3 Training Scaling Regimes

The training scaling wall appears when gradient communication time exceeds the
compute time that can overlap with it:

```
T_comm = 2 × ((p-1)/p) × model_size / BW_link
T_compute = FLOPs_per_step / (TFLOPS_per_GPU × N_GPUs)

Wall condition: T_comm > T_compute (can no longer hide comm behind compute)
```

For 7B model, p=8 GPUs, 25 GB/s per-rank PCIe bandwidth,
batch=32 sequences × seq_len=1024 = 32K tokens per step:

```
T_comm ≈ 2 × 0.875 × 14 GB / 25 GB/s ≈ 980 ms
T_compute ≈ 2 × 7×10⁹ × 32×1024 / (100 TFLOPS × 8) ≈ 570 ms

T_comm > T_compute → bandwidth-bound at 8 GPUs for this config
```

**The training wall for oneCCL on BMG/CRI is determined by:**
1. PCIe bandwidth to host staging buffers (same bottleneck as inference)
2. NIC aggregate bandwidth per node (100/200 GbE × number of NICs)
3. Model size and batch size

Unlike the inference wall (which is driven by the per-step host staging cost), the training wall
can be partially mitigated by **gradient compression**, **pipeline parallelism**
(which replaces large allreduces with small P2P activations), and **large batch sizes**
(more compute per byte of gradient). See [When to Use Which Collective — Training](03_when_to_use)
for the full breakdown.

### 6.4 Implications for BMG/CRI Training Deployments

```
Model    | Gradients | PCIe BW limit | Max GPUs before BW wall (est.)
---------|-----------|---------------|--------------------------------
1B  BF16 |    2 GB   |   25 GB/s     | ~32 GPUs (2 × 0.97 × 2 / 25 = 155 ms)
7B  BF16 |   14 GB   |   25 GB/s     | ~4–8 GPUs
13B BF16 |   26 GB   |   25 GB/s     | ~2–4 GPUs
70B BF16 |  140 GB   |   25 GB/s     | requires multi-node + PP to stay compute-bound
```

These are estimates; the actual wall depends on batch size and whether ZeRO sharding
is used. For large model training on BMG/CRI, pipeline parallelism (which avoids
large allreduces) matters more than algorithm selection.

---

## 7. MoE Expert-Parallel Communication

> **For MoE at 32-128+ EP nodes on Aurora:** this section models 8-node EP.
> For larger EP groups and the per-message posting overhead that dominates at
> scale, see [Aurora at Scale §5](aurora_large_scale).

DeepSeek-R1 (671B total, ~37B active per token, 256 experts) cannot fit on a
single node — it physically requires Expert Parallelism (EP) across multiple
nodes. This forces a different traffic pattern than Llama-3's TP allreduce, and it hits
the host staging wall through a different mechanism.

### 7.1 Multi-Node Requirements for Expert Parallelism

DeepSeek-R1's parameters (BF16): 671B × 2 bytes = 1.34 TB. Even with only
37B parameters active per token, the full expert table must be resident
somewhere — 256 experts × ~2.6 GB each. A single BMG/CRI node with 128 GB HBM
cannot hold the model. Expert parallelism distributes experts across nodes:
with 8 nodes, each node holds ~32 experts.

### 7.2 Dispatch and Combine Traffic

Each MoE layer executes two all-to-all collectives per forward pass:

```
1. Dispatch (all-to-all): each rank sends its tokens to the ranks
   holding the selected experts. Each token → top-k experts (k=8
   in DeepSeek-R1), possibly on different nodes.

2. Combine (all-to-all): after expert computation, results are
   sent back to the originating ranks.
```

For a batch of B tokens per rank, with hidden_dim=7168 and top-k=8:
```
Dispatch traffic per node-pair = B × k × hidden_dim × 2 bytes / N_nodes
                               = 128 × 8 × 7168 × 2 / 8 ≈ 1.8 MB per node-pair
```

(With 256 experts across 8 nodes = 32 per node, each token's k=8 experts
land on 8×(32/256) = 1 destination node on average — hence dividing by N_nodes.)

Each rank sends to 7 other nodes simultaneously. Total data staged through
host per rank per all-to-all:
```
Data per rank = 7 × 1.8 MB = 12.6 MB
```

### 7.3 Host-Staging Congestion Mechanism

The difference from Llama-3's allreduce:

**Llama-3 (allreduce):** data moves through host staging *sequentially* — one
hop at a time across log₂(N) steps. The bottleneck is latency accumulation
(α_staged × log₂(N)). PCIe is loaded by one transfer at a time per step.

**DeepSeek-R1 (all-to-all):** all N-1 rank-pairs stage *simultaneously*. Every
rank copies its outbound data to host DRAM at the same time, then all ranks'
NICs DMA from host DRAM at the same time. The bottleneck is aggregate host
DRAM bandwidth saturation (all tiles contend for the shared DDR5 bus).

On a PVC node with 12 ranks (6 GPUs × 2 tiles each) sharing 2 CPU sockets
(~307 GB/s DDR5 per socket, ~614 GB/s total host DRAM):
```
Aggregate D2H demand = 12 ranks × 12.6 MB = 151 MB simultaneous
Available host DRAM BW ≈ 614 GB/s (DDR5-4800, dual socket)
Time for D2H phase = 151 MB / 614 GB/s ≈ 0.25 ms

Aggregate NIC demand = 151 MB to wire
Available NIC BW = 8 NICs × 25 GB/s = 200 GB/s
Time for NIC phase = 151 MB / 200 GB/s ≈ 0.75 ms (host DRAM write completes
                                                     before NICs finish draining)
```

The actual bottleneck shifts between host DRAM bandwidth (D2H phase) and NIC
bandwidth (network phase) depending on load. Both are significantly slower
than GPU RDMA would be (0.5 ms total with direct NIC-to-GPU DMA at 200 GB/s).
All ranks contend for the shared host DRAM bus simultaneously, creating
bandwidth saturation that the sequential allreduce path never triggers — it
only loads one PCIe link at a time.

Total all-to-all time with host staging (D2H + network + H2D):
```
T_alltoall ≈ 0.25 ms (D2H, DDR5-limited) + 0.75 ms (wire, NIC-limited) + 0.25 ms (H2D) ≈ 1.25 ms
```

With direct GPU RDMA (no staging, NICs DMA from GPU BAR):
```
T_alltoall ≈ 151 MB / 200 GB/s ≈ 0.75 ms
```

Host staging inflates all-to-all by roughly **1.7× for MoE** in this model —
less dramatic than the allreduce case because at this data volume the NIC phase
dominates regardless. The qualitative point stands: the simultaneous-staging
pattern saturates the shared host DRAM bus in a way that sequential allreduce
does not, and the CPU-to-NIC posting overhead accumulates across all 11 peer
nodes in parallel. The mechanism is different: allreduce suffers per-hop
latency amplification; all-to-all suffers host DRAM bus contention.

### 7.4 Contrast Between Dense Decode and MoE Routing

| | Llama-3 70B (TP) | DeepSeek-R1 (EP) |
|---|---|---|
| Collective | allreduce | all-to-all |
| Message size | 16 KB | 1.8 MB per node-pair |
| Scaling behavior | O(log N) steps, sequential | O(N) peers, simultaneous |
| Staging bottleneck | Per-hop latency (α × log N) | Host DRAM BW saturation (simultaneous D2H) |
| Failure mode | Latency accumulation | Congestion/starvation |
| Wall location | N ≈ 16 nodes | N ≈ 4-8 nodes |
| With GPU RDMA | Wall → 64+ nodes | Wall → 32+ nodes |

The MoE wall appears earlier (4-8 nodes vs 16 nodes) because all-to-all's
simultaneous traffic pattern saturates the PCIe root complex at lower node
counts. Both walls share the same root cause — host staging — but the
all-to-all case is worse because it converts a bandwidth bottleneck into
a congestion bottleneck that algorithm tuning cannot remove (since the host-staging
copies remain regardless of algorithm choice).

See [Alltoall — MoE Expert Routing](../notebooks/03c_alltoall_moe) for
benchmark results and tuning guidance.

---

## 8. Aurora Measurement Context

The Aurora benchmark data (Ibeid et al., arXiv:2512.04291, Figure 14:
"Latency for MPI reduction operation for buffers located in GPU memory")
shows MPI_Allreduce latency with GPU-resident buffers scaling from 1 to
2048 nodes. Reading from Figure 14:

| Message | 1 node | 2048 nodes | Ratio |
|---|---|---|---|
| 8 B | ~15 us | ~250 us | 16.7x |
| 64 KB | ~50 us | ~280 us | 5.6x |

The paper notes: "Less than linear latency growth is observed, which is
typical for a recursive-doubling tree algorithm. A switch from a ring
algorithm to a tree algorithm is clearly seen on the curves."

For 8 B (effectively zero bandwidth term), latency scales 16.7x across 11
effective inter-node doubling steps (log₂(2048) = 11, plus intra-node phases).
Theoretical log₂ scaling without overhead would predict ~3x (11 steps vs ~4
equivalent intra-node steps). The 16.7x measured result vs 3x theoretical
ratio — approximately 5x excess — is consistent with a 5x per-step overhead
multiplier from host staging on top of network latency.

For 64 KB, the bandwidth term amortizes the per-step fixed overhead, producing
better scaling (5.6x). This matches the theory: bounce buffer overhead is
most damaging for small messages, which are the dominant case in decode.

The HiCCL paper (Hidayetoglu et al., arXiv 2408.05962) reported a **12.1x
geometric mean improvement** over oneCCL across allreduce, broadcast, and
all-to-all on 4 Aurora nodes. HiCCL's primary mechanism was topology-aware
decomposition that concentrates staging traffic on NIC-proximate tiles and
avoids unnecessary PCIe crossings. This indicates that topology-aware handling
of staged traffic can materially improve performance even at small scale.

A follow-on study, CoCoDiff (Ma et al., arXiv:2604.14561, 2025), reported
**3.6× average (8.4× peak) speedup** over the oneCCL baseline when scaling
distributed diffusion-transformer inference across 96 Intel GPU tiles (8 nodes)
on Aurora. CoCoDiff's Tile-Aware Parallel All-to-all (TAPA) achieves this
by aligning collective patterns with Aurora's two-tier (Xe Link intra-node,
Slingshot inter-node) topology — the same reason the `topo` hierarchical
algorithm in §3.3 reduces inter-node message count. That 3–8× gap at only
8 nodes is consistent with the model in §3.1.

---

## 9. Comparison with GPU Direct RDMA

On NVIDIA hardware, NIC-to-GPU DMA is enabled by default via `nvidia-peermem`.
The GPU kernel cannot autonomously post RDMA operations (that requires
GDRCopy/NVSHMEM), but the NIC reads/writes GPU memory directly. Per-step
latency:

```
α_NVIDIA = α_NIC_GPU_DMA_out + α_network + α_NIC_GPU_DMA_in
         ≈ 0.5 + 1.5 + 0.5 = 2.5 us
```

On Intel PVC in production (OFI, no HMEM):

```
α_Intel = α_D2H + α_host_task + α_OFI + α_network + α_wait + α_H2D
        ≈ 2 + 0.5 + 1.5 + 1.5 + 0.5 + 2 = 8 us
```

| Path | α per step | 2048-node recursive doubling (11 steps) |
|---|---|---|
| NVIDIA GPUDirect RDMA | ~2.5 us | ~28 us |
| Intel HMEM (`CCL_ATL_HMEM=1`, if `FI_HMEM_ZE` confirmed active) | ~4 us | ~44 us |
| Intel staging (current verified baseline on Aurora) | ~8 us | ~88 us base + scheduler overhead |
| Measured Aurora | — | ~250 us |

The gap between the 88 us model and the 250 us measurement is scheduler
overhead in oneCCL's sched/entry framework, SYCL event graph evaluation,
and the intra-node Xe Link phases.

The CommBench micro-benchmark (Hidayetoglu et al., ICS 2024) independently
measured **~8 µs per-step allreduce latency** on Aurora for small messages
with the (n, 12, 12) tile configuration, corroborating the α_staged ≈ 8 µs
estimate in §2.3 and §3.1.

The Intel HMEM path (`CCL_ATL_HMEM=1`) would eliminate the D2H and H2D
copies, with the CPU still calling `fi_tsendmsg()` and data going GPU → NIC
→ wire → NIC → GPU via PCIe BAR mapping (Linux dmabuf). The hardware, xe
driver, and libfabric util-layer (`src/hmem_ze.c`, `#if HAVE_ZE`) are all
capable — Allcock et al. confirm GPU Direct RDMA via dma-buf P2P DMA works
through MPICH on Aurora, demonstrating that the platform can support
GPU-memory DMA through this mechanism. Whether `CCL_ATL_HMEM=1` activates
this path in oneCCL depends on its ATL probe succeeding against the deployed
libfabric provider. The ~4 µs figure is contingent on the HMEM
path being confirmed active. The remaining
gap vs NVIDIA after that would be CPU-still-posts overhead (~1.5 µs/step),
which UALink and GPU-initiated RDMA are designed to close.

---

## 10. Deployment Guidance

If you are debugging a performance regression or deploying on a BMG/CRI system
today, these are the flags to try in order of impact. None eliminate host
staging, but together they recover 40–60% of the overhead in most decode
workloads.

### 10.1 Runtime Configuration Options

**1. `async_op=True` on every collective call**

```python
handle = dist.all_reduce(tensor, async_op=True)
# ... run next layer's compute here ...
handle.wait()
```

This allows the next layer's compute to proceed while the current layer's
allreduce is in flight — the most direct way to reduce visible stall time. At N=8 nodes
(T_comm=40 µs), the overlap window is limited to non-dependent inter-layer
work (~5-15 µs of RMSNorm, RoPE, etc.), but this still recovers 5-15 µs
per allreduce that would otherwise be pure stall.

**2. `CCL_ATL_HMEM=1` (verify before relying on it)**

```bash
export CCL_ATL_HMEM=1
export CCL_LOG_LEVEL=info  # verify with: grep "use_hmem: 1" startup log
```

Eliminates the D2H and H2D memcpy steps entirely (see §2.0), cutting per-step
latency from ~8 µs to ~4 µs. At N=8 nodes: T_comm drops from 40 µs to ~22 µs.

The underlying stack is demonstrated: xe driver exports GPU BAR via dmabuf,
libfabric util-layer `src/hmem_ze.c` has a complete `FI_HMEM_ZE`
implementation, Cassini NIC can DMA from dmabuf-registered memory, and MPICH
does GPU Direct RDMA via this path on Aurora (Allcock et al.,
arXiv:2509.08207), demonstrating that the platform can support GPU-memory
DMA through this mechanism. Whether oneCCL activates HMEM depends on its ATL
probe succeeding against the deployed libfabric provider — the `HAVE_ZE`
compile flag is the most commonly cited gate, but oneCCL's probe path may
have additional requirements. The CXI provider's `force_ze_hmem_support`
environment variable may also be needed.

**Always confirm with `"use_hmem: 1"` in the startup log.** If it does not
appear, oneCCL silently fell back to host staging and the flag had no effect.

**3. NUMA pinning**

```bash
export I_MPI_PIN_DOMAIN=socket
```

Ensures each MPI rank's host staging buffer is on the NUMA node that shares
a PCIe root complex with that rank's GPU. Reduces D2H/H2D latency from the
contended 3 µs (cross-socket path) toward the uncontended 1 µs (local socket).
Most effective on 2-socket nodes where half the tiles would otherwise stage
across the interconnect.

**4. Worker thread count**

```bash
export CCL_WORKER_COUNT=1  # for low-rate decode (default, leave it)
export CCL_WORKER_COUNT=2  # for high-rate prefill or training with many concurrent collectives
```

Increasing from 1 to 2 workers helps when queue depth exceeds 1 (see §4.4).
Do not set higher than 2 without profiling — at 4+ workers, L3 and PCIe
contention on the staging buffers increases per-operation latency.

**5. `CCL_ALLREDUCE_SCALEOUT`**

```bash
export CCL_ALLREDUCE_SCALEOUT=ring  # large messages only (prefill, training)
# leave unset for decode (MPI selects a log-step algorithm internally)
```

For large-message allreduces (> 2 MB BF16, common in prefill and training),
ring is more bandwidth-efficient than the default. For decode (16 KB messages),
leaving this unset is correct — the selector already picks recursive doubling.

### 10.2 Expected Effect on Decode Latency

For Llama-3 70B decode at N=8 nodes (64 tiles, T_compute ≈ 256 µs/layer):

| Config | T_comm per allreduce | Comm fraction (2 allreduces/layer) |
|---|---|---|
| Default (staging, no async) | 40 µs | 80 / 336 = 24% |
| + `async_op=True` | ~25 µs visible (15 µs hidden) | 50 / 306 = 16% |
| + NUMA pinning | ~18 µs total | 36 / 292 = 12% |
| `CCL_ATL_HMEM=1` (if `FI_HMEM_ZE` confirmed active) | ~22 µs | 44 / 300 = 15% |
| Hypothetical GPU RDMA | ~13 µs total | 26 / 282 = 9% |

At N=8 nodes, communication overhead is 12-24% — significant but not yet
dominant. At N=16+ nodes, T_compute drops to ~128 µs and communication
fraction reaches 44%+. For Llama-3 70B decode on BMG/CRI, the model
suggests 4–8 nodes as a reasonable operating range; beyond 16 nodes,
allreduce growth outpaces the compute savings from adding ranks.

### 10.3 Conditions That Change the Scaling Model

Two things would reduce the bounce buffer cost:

- **HMEM path confirmed on Aurora**: the hardware, xe driver, and libfabric
  util-layer (`src/hmem_ze.c`) are all capable. The CXI provider routes through
  the shared `FI_HMEM_ZE` dispatch and has a `force_ze_hmem_support` knob. The
  remaining step is verifying that Aurora's deployed libfabric, driver, and
  kernel stack have `FI_HMEM_ZE` active — a deployment question, not a code gap.
  If confirmed, `CCL_ATL_HMEM=1` would bring N=8-node allreduce from ~40 µs to
  ~22 µs per op (~45% reduction) at no hardware cost.

- **GPU-initiated RDMA on future Intel hardware**: would eliminate the CPU
  from the critical post path entirely, cutting α from ~4 µs (HMEM) to ~2.5 µs
  (GPU-posted NIC DMA). This approaches NVIDIA GPUDirect parity and would push
  the practical scaling wall from ~16 nodes to approximately 64+ nodes.

Until one of those changes, the table above is the expected operating range.

### 10.4 Mitigation Limits

| Strategy | Mechanism | Effect | Ceiling |
|---|---|---|---|
| `async_op=True` | Overlap collective with next layer's compute | Hides latency up to T_compute | No benefit when T_comm > T_compute |
| `CCL_ALLREDUCE_SCALEOUT=ring` | Bandwidth-efficient for large messages | Better bandwidth, worse latency | Harmful for small-message decode |
| `CCL_ATL_HMEM=1` | NIC reads/writes GPU memory directly | Eliminates D2H/H2D data copies | Requires `FI_HMEM_ZE` in deployed libfabric (build + runtime); verify `use_hmem: 1` in startup log |
| NUMA pinning | Reduces cross-socket PCIe hops | Reduces per-tile D2H latency | Only fixes intra-node PCIe routing |
| `CCL_WORKER_COUNT` increase | More worker threads | Reduces serialization at high rates | L3/PCIe contention at high counts |
| `TMP_BUF` | Pre-copies buffer for async semantics | Frees user buffer earlier | Adds 2 extra copies |

None of these remove the staging overhead from the data path. The two changes
that would are:

1. **Aurora libfabric `HAVE_ZE` build + HMEM validation at scale**: hardware,
   xe driver, and libfabric util-layer are all ready. Requires confirming the
   deployed libfabric includes `HAVE_ZE` (a build/system configuration step)
   and validating `CCL_ATL_HMEM=1` at production scale. Once confirmed,
   eliminates D2H/H2D data movement; leaves ~1.5 µs CPU-post overhead per
   step. Target latency: ~44 µs at 2048 nodes vs ~250 µs today.

2. **GPU-initiated RDMA (future hardware)**: eliminates CPU from critical path
   entirely. Target: ~28 us at 2048 nodes, parity with NVIDIA production.

3. **UALink for intra-rack collectives**: bypasses host staging for rack-scale
   all-reduce, limiting staging overhead to inter-rack traffic only.

---

## 11. Summary

The host bounce buffer in oneCCL's default configuration adds approximately
6-8 us of per-step overhead to every inter-node collective on this path.
The code shows this indirectly through the scale-out structure:

- `allreduce_scaleout_sycl.cpp` line 119: OFI forces `copy_to_host=true`
  unconditionally, regardless of `CCL_SYCL_ENABLE_DIRECT_GPU_RDMA`
- `allreduce_scaleout_sycl.cpp` and related paths: use `host_task` to submit
  blocking communication calls, which creates a massive scheduler event-dependency
  bottleneck within the SYCL runtime.
- `coll_util.cpp` (scaleout path): `!enable_hmem` gates D2H copy for every
  inter-node allreduce, allgather, reduce-scatter, all-to-all, and reduce

For recursive doubling at 2048 nodes (11 steps):

```
Overhead from staging: 11 * 6 us = 66 us (modeled lower bound)
Measured total: ~250 us
NVIDIA GPUDirect equivalent: ~28 us
```

The alpha-beta model in Section 5 — using PVC tile HBM bandwidth from Intel
product specifications, memory-bound per-layer weight streaming time at
batch=1, and a staged α of 8-12 µs per step from the source analysis in
Section 2 — estimates the scaling crossover at approximately 8-16 nodes for
decode inference with current oneCCL defaults. Beyond that point, allreduce
latency exceeds per-layer compute time and tensor parallelism degrades
efficiency faster than it improves throughput. The Aurora data from Ibeid et
al. (arXiv:2512.04291) is consistent with this model but does not pin the
workload-specific crossover; measuring it requires end-to-end decode profiling
on the target model and node count.

---

## References

### oneCCL Source

- `src/coll/algorithms/allreduce/sycl/allreduce_scaleout_sycl.cpp`:
  lines 29-100 (host_task orchestration and blocking wait behavior),
  lines 116-121 (OFI forces copy_to_host=true)
- `src/coll/coll_util.cpp`: scaleout path, enable_hmem gate, host buffer allocation
- `src/coll/selection/selector_allreduce.cpp`: algorithm selection by message size
- `src/coll/selection/selector.hpp`: CCL_ALLREDUCE_SHORT_MSG_SIZE=8192,
  CCL_ALLREDUCE_MEDIUM_MSG_SIZE=1048576
- `src/coll/algorithms/allreduce/allreduce.cpp`:
  lines 525-639 (recursive doubling implementation),
  lines 427-523 (ring with limited overlap logic)

### Benchmarks and Literature

- Ibeid et al., "Scaling MPI Applications on Aurora" (arXiv:2512.04291, Dec 2025)
  — Aurora allreduce scaling data (Fig 14), NIC effective bandwidth (Fig 12),
  PCIe Gen4→Gen5 conversion overhead, recursive-doubling/tree-like behavior indicated
  by published scaling curves
- Allcock et al., "Aurora: Architecting Argonne's First Exascale Supercomputer"
  (arXiv:2509.08207, Sep 2025) — ECB topology diagram (Fig 4), PCIe Gen5 x16
  GPU→CPU (64 GB/s), PCIe Gen4 NIC path (32 GB/s), GPU Direct RDMA via dmabuf,
  DDR5 8-channel memory per socket, Slingshot-11 Cassini 200 Gbps NIC
- Goto et al., "Sustaining Exascale Performance: Lessons from HPL and HPL-MxP
  on Aurora" (arXiv:2604.09517, 2026) — confirms "PCIe switches
  that fan out Gen5 x16 lanes to Gen4 x16 endpoints" for NIC-facing ports
- Hidayetoglu et al., "HiCCL: A Hierarchical Collective Communication Library"
  (arXiv:2408.05962, IPDPS 2025) — 12.1× improvement over oneCCL at 4 nodes,
  topology-aware staging reduction
- Hidayetoglu et al., "CommBench: Micro-benchmarking Hierarchical Networks with
  Multi-GPU, Multi-NIC Nodes" (ICS 2024) — ~8 µs allreduce latency on Aurora,
  (n, 12, 12) tile configuration
- Ma et al., "CoCoDiff: Optimizing Collective Communications for Distributed
  Diffusion Transformer Inference Under Ulysses Sequence Parallelism"
  (arXiv:2604.14561, 2025) — 3.6× average / 8.4× peak speedup over oneCCL
  baseline at 96 tiles (8 nodes) on Aurora; confirms staging as bottleneck
- Thakur et al., "Optimization of Collective Communication Operations in MPICH"
  (IJHPCA, 2005) — latency/bandwidth model for ring and recursive doubling
- Chan et al., "Collective Communication: Theory, Practice, and Experience"
  (Concurrency and Computation, 2007) — alpha-beta model derivations
- Rabenseifner, "Optimization of Collective Reduction Operations"
  (ICCS 2004) — reduce-scatter + allgather decomposition
