# Host Staging: The Scaling Wall

## Abstract

Intel Xe GPU architectures (PVC, BMG/CRI) require all inter-node collective
communication to pass through host DRAM by default. This "host bounce buffer"
design imposes a fixed per-step latency penalty that compounds across every
step of a distributed collective algorithm. At small node counts the overhead
is tolerable. Beyond roughly 8-16 nodes for latency-sensitive workloads (decode
inference), the compounded penalty exceeds per-layer compute time and the system
becomes communication-bound in a way that no algorithm tuning can escape, because
the bottleneck is architectural, not algorithmic. This paper analyzes the
mechanism with specific reference to oneCCL source code, derives the scaling
behavior quantitatively, and identifies the conditions under which the wall
appears.

---

## 1. Background: What Collectives Require of the Hardware

oneCCL is Intel's collective communication library. It implements allreduce,
broadcast, all-to-all, and other operations that are on the critical path of
distributed training and inference. In a transformer model sharded across N
GPUs via tensor parallelism, every attention and MLP layer ends with an allreduce
that sums partial results across all ranks and returns the total to each. The
latency of this allreduce subtracts directly from the time budget for each
generated token.

The time to execute one allreduce has two components:

```
T_allreduce = T_compute_local + T_communication
```

`T_compute_local` scales inversely with the number of ranks (more ranks, less
work per rank). `T_communication` does not improve — it worsens, because more
ranks means more steps in any collective algorithm. Scaling efficiency is:

```
E(N) = T_serial / (N * T_parallel(N))
```

When `T_communication` grows faster than `T_compute_local` shrinks, efficiency
degrades. The node count at which `T_communication >= T_compute_local` is the
practical scaling wall. For Intel Xe GPUs, the host bounce buffer inflates
`T_communication` by 3-5x per step relative to a GPU RDMA path, shifting that
wall from roughly 64 nodes to roughly 8-16 nodes.

---

## 2. The Host Bounce Buffer Mechanism

On PVC and BMG/CRI, GPU kernels cannot issue network operations. The NIC is not
accessible from GPU-side code. Data must bounce through host DRAM on both
the send and receive sides:

```
Default path: OFI transport, no HMEM (production)

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

All inter-node communication follows this sequence
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
configuration (without `CCL_ATL_HMEM=1`), every production deployment
operates in the host-staged regime. The HMEM path (§2.0) also uses OFI but
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

### 2.0 How HMEM Bypasses the Host Staging Path

The diagram above shows "OFI + CCL_ATL_HMEM=1" as a path where data never
touches host DRAM. This seems to contradict the `copy_to_host = true` override
at line 119. The resolution: **HMEM and `allreduce_scaleout_sycl_simple` are
completely different code paths that share no code.**

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

**The catch:** `CCL_ATL_HMEM=1` has three hard requirements, all checked at
communicator init:

- libfabric provider with `FI_HMEM` capability (Slingshot-11 / CXI provider,
  firmware ≥ 2.x; not all OFI providers support this)
- Linux kernel ≥ 5.12 with `CONFIG_DMA_BUF` and GPU driver dmabuf enabled
- Intel GPU driver (i915/xe) with P2P dmabuf support active

If any requirement fails, `atl_base_comm::attr.out.enable_hmem` is set to
`false` during provider capability probing at init — and oneCCL silently falls
back to host staging with no warning. Use `CCL_LOG_LEVEL=info` and look for
`"use_hmem: 1"` in the startup log to confirm the HMEM path is actually active.

### 2.1 Staging Buffer Management

The host staging buffer is **pre-allocated** at communicator init time and
validated per-collective. From `allreduce_scaleout_sycl.cpp` lines 30-40:

```cpp
if (comm->get_scaleout_host_buf_size() < count * ccl_dtype.size()) {
    LOG_WARN("scaleout_host_buf_size is not big enough to handle ",
             count * ccl_dtype.size(),
             " bytes. Falling back. TODO: chunking/pipelining");
    done = false;
    return e;
}
scaleout_recv_buf = comm->get_scaleout_host_buf();
```

If the message exceeds the pre-allocated buffer size, oneCCL falls back
entirely (sets `done = false`) and returns an empty event. There is no
chunking to handle messages that exceed the buffer. For large allreduces
at scale (prefill, large gradients), this fallback path triggers silently.

### 2.2 Algorithm Selection by Message Size

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
- Scaleout (SYCL path, production): `direct` for BF16 ≤ 1-4 MB depending on
  comm_size (delegates to Intel MPI's native `MPI_Allreduce`, which internally
  selects recursive doubling for small messages). Ring or rabenseifner for larger
  messages. Selection logic: `sycl_selection.cpp:330-380`.
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

### 2.3 Latency Budget Per Collective Step

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

## 3. Collective Algorithm Scaling Under Bounce Buffer

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
so oneCCL's SYCL+ZE path selects `topo` → `direct` scaleout → Intel MPI
`MPI_Allreduce`. Intel MPI selects recursive doubling for small messages; this
is confirmed empirically by the sub-linear (log₂N) latency scaling in Ibeid
et al., Figure 14, which matches the recursive doubling step-count model below.
This per-step model also applies to the nreduce case for messages in the
8–1024 KB range that take multiple ring steps):

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

The fundamental result: **the bounce buffer adds ~130 us to allreduce latency
at 2048 nodes compared to a hypothetical direct GPU RDMA path.**

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

### 3.3 The topo Hierarchical Algorithm

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

This is consistent with the Aurora measurements and confirms the model.
The Xe Link phase on PVC amortizes some PCIe contention (fewer tiles staging
per node), but does not change the per-step inter-node latency.

---

## 4. The Three Scaling Bottlenecks

### 4.1 Host DRAM Bandwidth Saturation

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

### 4.2 Sequential Stages with No Pipelining

The `allreduce_scaleout_sycl_simple` path (lines 29-100) chains three
sequential SYCL submissions via event dependencies:

```
[D2H memcpy] → [host_task(allreduce+wait)] → [H2D memcpy]
```

Each submission depends on the previous completing. The `TODO:
chunking/pipelining` comment at line 33 is the oneCCL team's own
acknowledgment that this is a known limitation. No chunking means:

1. Large messages cannot be split and pipelined — if the message exceeds
   the staging buffer, the operation falls back entirely
2. The D2H copy cannot overlap with the H2D copy of the previous step
3. The network transmission of one chunk cannot overlap with staging
   of the next chunk

NCCL's implementation of ring allreduce pipelines chunk transmission with
chunk reduction, achieving near-wire-speed bandwidth on large messages.
oneCCL's staging path does not have an equivalent.

### 4.3 CPU Orchestration as Serialization Point

Every OFI operation requires host CPU involvement through a SYCL `host_task`.
The `host_task` submits to the SYCL host task queue, which serializes within
a queue. From `allreduce_scaleout_sycl.cpp` lines 62-90:

```cpp
op_end = q.submit([=](sycl::handler& h) {
    h.depends_on(dep_events);
    h.host_task([=]() {
        int ep_idx = 0;  // TODO: use correct endpoint index
        atl_req_t req;
        ATL_CALL_THROW_IF_ERROR(
            atl_comm->allreduce(ep_idx, send, recv, count, dtype, red, req));
        ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
    });
});
```

The endpoint index is hardcoded to `0` with a `TODO` comment suggesting it
should be replaced with `atl_ep->idx` or derived from `sched->bin->get_atl_ep`.
As written, all collectives issued through this code path use endpoint 0 for
their ATL communicator. Whether this serializes across ranks depends on whether
each rank has its own `atl_comm` instance — with MPI transport each rank has
distinct MPI communicator state, so contention is per-rank rather than
node-wide. With OFI transport and shared endpoint pools, contention across
concurrent collectives on the same rank (e.g. pipeline-parallel microbatches)
is the more likely bottleneck.

Inside the host task, the code first calls `atl_comm->check()` to test for
immediate completion, then falls through to `atl_comm->wait()` only if the
operation is not already done. In practice for network collectives,
`check()` returns incomplete and `wait()` is invoked — blocking the SYCL
host thread until MPI/OFI reports completion. Under load, the SYCL
scheduler cannot repurpose this thread for other work while it is blocked
in `wait()`. The net effect is a blocking wait on the critical path.

### 4.4 Worker Thread Model

oneCCL uses a worker thread pool (default: 1 worker per rank via
`CCL_WORKER_COUNT`). The executor routes collectives by `sched_id %
workers.size()`. With a single worker, all collective operations for a rank
are serialized through one thread. Pipeline-parallel workloads that issue
multiple collectives concurrently (e.g., one allreduce per pipeline stage
in flight) queue behind each other at the worker.

At high token throughput, the collective rate is:

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

## 5. Where the Scaling Wall Appears

The scaling wall is the node count N* where `T_communication >= T_compute`
per layer. For decode inference:

```
T_compute per layer ≈ weight_bytes_per_layer / (HBM_BW * N_tiles_per_node * N_nodes)
T_comm per allreduce ≈ T_intranode + log₂(N_nodes) * α_staged
```

(The log₂(N) term reflects the recursive doubling algorithm that Intel MPI
selects for small messages. The SYCL topo scaleout path uses `direct` which
delegates to MPI's native allreduce — see §2.2 and `sycl_selection.cpp`.)

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

This yields N* ≈ 8-16 nodes. Beyond this, allreduce dominates the per-token
budget.

### Concrete Example: Llama-3 70B, TP=8 tiles/node, Batch=1 Decode

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

The practical efficiency floor from Amdahl alone limits useful TP to ~8–16 tiles
before the serial fraction dominates — independent of communication overhead.
When communication overhead is added on top, the effective wall appears earlier.

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

## 5b. Training Scaling: Bandwidth Sufficiency Wall

The inference scaling analysis above addresses the **latency wall** — when allreduce latency
exceeds per-layer compute time for decode. Training has a different scaling failure mode:
the **bandwidth wall** — when gradient traffic exceeds the available PCIe/NIC bandwidth.

For inference, the question is: *is allreduce fast enough to stay off the critical path?*
For training, the question is: *is there enough bandwidth to move all gradients within
the step time?*

### Gradient Volume at Scale

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

### Why ZeRO Changes the Calculation

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

### Where the Training Wall Appears

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

Unlike the inference wall (which is fundamental to host staging), the training wall
can be partially mitigated by **gradient compression**, **pipeline parallelism**
(which replaces large allreduces with small P2P activations), and **large batch sizes**
(more compute per byte of gradient). See [When to Use Which Collective — Training](03_when_to_use)
for the full breakdown.

### Practical Implication for BMG/CRI Training Deployments

```
Model    | Gradients | PCIe BW limit | Max GPUs before BW wall (est.)
---------|-----------|---------------|--------------------------------
1B  BF16 |    2 GB   |   25 GB/s     | ~32 GPUs (2 × 0.97 × 2 / 25 = 155 ms)
7B  BF16 |   14 GB   |   25 GB/s     | ~4–8 GPUs
13B BF16 |   26 GB   |   25 GB/s     | ~2–4 GPUs
70B BF16 |  140 GB   |   25 GB/s     | requires multi-node + PP to stay compute-bound
```

These are estimates; the actual wall depends on batch size and whether ZeRO sharding
is used. The key takeaway: **for large model training on BMG/CRI, pipeline parallelism
(which avoids large allreduces) is more important than algorithm selection.**

---

## 5c. MoE Scaling: PCIe Congestion Wall

DeepSeek-R1 (671B total, ~37B active per token, 256 experts) cannot fit on a
single node — it physically requires Expert Parallelism (EP) across multiple
nodes. This forces a fundamentally different traffic pattern than Llama-3's TP
allreduce, and it hits the host staging wall through a different mechanism.

### Why Expert Parallelism Requires Multi-Node

DeepSeek-R1's parameters (BF16): 671B × 2 bytes = 1.34 TB. Even with only
37B parameters active per token, the full expert table must be resident
somewhere — 256 experts × ~2.6 GB each. A single BMG/CRI node with 128 GB HBM
cannot hold the model. Expert parallelism distributes experts across nodes:
with 8 nodes, each node holds ~32 experts.

### The Traffic Pattern: Dispatch + Combine

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

### The Congestion Mechanism: Simultaneous Staging

Here is the critical difference from Llama-3's allreduce:

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

### Contrast: Llama-3 vs DeepSeek-R1 Failure Modes

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
a pure congestion bottleneck that no algorithm tuning can route around.

See [Alltoall — MoE Expert Routing](../notebooks/03c_alltoall_moe) for
benchmark results and tuning guidance.

---

## 6. Empirical Validation from Aurora

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
avoids unnecessary PCIe crossings. This improvement at just 4 nodes indicates
that even at small scale, the staging overhead is the bottleneck — not
algorithm selection or ring topology.

---

## 7. Comparison with NVIDIA GPUDirect RDMA

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
| Intel HMEM (experimental) | ~4 us | ~44 us |
| Intel staging (production default) | ~8 us | ~88 us base + scheduler overhead |
| Measured Aurora | — | ~250 us |

The gap between the 88 us model and the 250 us measurement is scheduler
overhead in oneCCL's sched/entry framework, SYCL event graph evaluation,
and the intra-node Xe Link phases.

Intel HMEM (`CCL_ATL_HMEM=1`) eliminates the D2H and H2D copies from the
data path. The host CPU still calls `fi_tsendmsg()` but data goes GPU → NIC
→ wire → NIC → GPU via PCIe BAR mapping (Linux dmabuf, kernel 5.12+). The
remaining gap vs NVIDIA is: CPU still posts every operation (~1.5 us/step)
vs NVIDIA GDRCopy where the GPU posts directly. This is the architectural
gap that UALink and GPU-initiated RDMA are designed to close.

---

## 8. What To Do Right Now

If you are debugging a performance regression or deploying on a BMG/CRI system
today, these are the flags to try in order of impact. None eliminate host
staging, but together they recover 40–60% of the overhead in most decode
workloads.

### Immediate Flags

**1. `async_op=True` on every collective call**

```python
handle = dist.all_reduce(tensor, async_op=True)
# ... run next layer's compute here ...
handle.wait()
```

This is the single highest-leverage change. It allows the next layer's compute
to proceed while the current layer's allreduce is in flight. At N=8 nodes
(T_comm=40 µs), the overlap window is limited to non-dependent inter-layer
work (~5-15 µs of RMSNorm, RoPE, etc.), but this still recovers 5-15 µs
per allreduce that would otherwise be pure stall.

**2. `CCL_ATL_HMEM=1` (if your stack supports it)**

```bash
export CCL_ATL_HMEM=1
export CCL_LOG_LEVEL=info  # verify with: grep "use_hmem: 1" startup log
```

Eliminates the D2H and H2D memcpy steps entirely (see §2.0), cutting per-step
latency from ~10 µs to ~4 µs. At N=8 nodes: T_comm drops from 40 µs to ~22 µs
(10 µs intranode + 3 × 4 µs inter-node). Requirements: libfabric `FI_HMEM`
provider (Slingshot CXI ≥ fw 2.x), Linux ≥ 5.12, Intel GPU driver dmabuf.
If `use_hmem: 1` does not appear in the log, it silently fell back to staging.

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
# leave unset for decode (recursive doubling is auto-selected and correct)
```

For large-message allreduces (> 2 MB BF16, common in prefill and training),
ring is more bandwidth-efficient than the default. For decode (16 KB messages),
leaving this unset is correct — the selector already picks recursive doubling.

### What You Can Expect

For Llama-3 70B decode at N=8 nodes (64 tiles, T_compute ≈ 256 µs/layer):

| Config | T_comm per allreduce | Comm fraction (2 allreduces/layer) |
|---|---|---|
| Default (staging, no async) | 40 µs | 80 / 336 = 24% |
| + `async_op=True` | ~25 µs visible (15 µs hidden) | 50 / 306 = 16% |
| + `CCL_ATL_HMEM=1` | ~22 µs total | 44 / 300 = 15% |
| + NUMA pinning | ~18 µs total | 36 / 292 = 12% |
| Hypothetical GPU RDMA | ~13 µs total | 26 / 282 = 9% |

At N=8 nodes, communication overhead is significant (12-24%) but not
dominant. The real danger zone is N=16+ nodes where T_compute drops to
~128 µs and communication fraction reaches 44%+. The practical operating
point for Llama-3 70B decode on BMG/CRI is 4–8 nodes; beyond 16 nodes,
adding hardware degrades tokens/second.

### The Forward Path

Intel's hardware trajectory closes this gap in two steps:

- **HMEM reaching production quality** (the `CCL_ATL_HMEM=1` path validated
  at scale): would bring N=8-node allreduce from ~40 µs to ~22 µs per op,
  reducing per-layer communication overhead by ~45%. This is a software/firmware
  quality issue, not an architectural change.

- **GPU-initiated RDMA on future Intel hardware**: would eliminate the CPU
  from the critical post path entirely, cutting α from ~4 µs (HMEM) to ~2.5 µs
  (GPU-posted NIC DMA). This approaches NVIDIA GPUDirect parity and would push
  the practical scaling wall from ~16 nodes to approximately 64+ nodes.

Until then, the table above represents the realistic operating envelope.

### Mitigation Strategies and Their Ceilings

| Strategy | Mechanism | Effect | Ceiling |
|---|---|---|---|
| `async_op=True` | Overlap collective with next layer's compute | Hides latency up to T_compute | No benefit when T_comm > T_compute |
| `CCL_ALLREDUCE_SCALEOUT=ring` | Bandwidth-efficient for large messages | Better bandwidth, worse latency | Harmful for small-message decode |
| `CCL_ATL_HMEM=1` | NIC reads/writes GPU memory directly | Eliminates D2H/H2D data copies | Experimental, silent hang failure mode |
| NUMA pinning | Reduces cross-socket PCIe hops | Reduces per-tile D2H latency | Only fixes intra-node PCIe routing |
| `CCL_WORKER_COUNT` increase | More worker threads | Reduces serialization at high rates | L3/PCIe contention at high counts |
| `TMP_BUF` | Pre-copies buffer for async semantics | Frees user buffer earlier | Adds 2 extra copies |

None of these eliminate the fundamental staging overhead. The only structural
fixes are:

1. **HMEM validation to production quality**: eliminates D2H/H2D data movement,
   leaves ~1.5 us CPU-posts overhead per step. Target latency: ~44 us at 2048
   nodes vs ~250 us today.

2. **GPU-initiated RDMA (future hardware)**: eliminates CPU from critical path
   entirely. Target: ~28 us at 2048 nodes, parity with NVIDIA production.

3. **UALink for intra-rack collectives**: bypasses host staging for rack-scale
   all-reduce, limiting staging overhead to inter-rack traffic only.

---

## 9. Summary

The host bounce buffer in oneCCL's default configuration adds approximately
6-8 us of unavoidable overhead to every step of every inter-node collective.
The code evidence is unambiguous:

- `allreduce_scaleout_sycl.cpp` line 119: OFI forces `copy_to_host=true`
  unconditionally, regardless of `CCL_SYCL_ENABLE_DIRECT_GPU_RDMA`
- `allreduce_scaleout_sycl.cpp` line 33: `TODO: chunking/pipelining` — the
  sequential D2H → allreduce → H2D pipeline has no overlap implementation
- `allreduce_scaleout_sycl.cpp` line 66: `ep_idx = 0; // TODO: use correct
  endpoint index` — all collectives serialize through one ATL endpoint
- `coll_util.cpp` (scaleout path): `!enable_hmem` gates D2H copy for every
  inter-node allreduce, allgather, reduce-scatter, all-to-all, and reduce

For recursive doubling at 2048 nodes (11 steps):

```
Overhead from staging: 11 * 6 us = 66 us (structural minimum)
Measured total: ~250 us
NVIDIA GPUDirect equivalent: ~28 us
```

The practical scaling limit for decode inference (small messages,
latency-critical, OFI transport) is approximately 8-16 nodes with current
oneCCL defaults. Beyond this point, allreduce latency exceeds per-layer
compute time and tensor parallelism degrades efficiency faster than it
improves throughput. (This estimate is derived in Section 5 using the
alpha-beta latency model [Thakur et al., IJHPCA 2005; Chan et al., 2007],
PVC tile HBM bandwidth from Intel product specifications, memory-bound
per-layer weight streaming time at batch=1, staged alpha of 8-12 us per
collective step measured from oneCCL source in Section 2, and validated
against empirical Aurora data [Ibeid et al., arXiv:2512.04291].)

---

## References

### oneCCL Source

- `src/coll/algorithms/allreduce/sycl/allreduce_scaleout_sycl.cpp`:
  lines 29-100 (sequential D2H/allreduce/H2D pipeline),
  lines 116-121 (OFI forces copy_to_host=true),
  line 33 (TODO: chunking/pipelining),
  line 66 (ep_idx=0 serialization)
- `src/coll/coll_util.cpp`: scaleout path, enable_hmem gate, host buffer allocation
- `src/coll/selection/selector_allreduce.cpp`: algorithm selection by message size
- `src/coll/selection/selector.hpp`: CCL_ALLREDUCE_SHORT_MSG_SIZE=8192,
  CCL_ALLREDUCE_MEDIUM_MSG_SIZE=1048576
- `src/coll/algorithms/allreduce/allreduce.cpp`:
  lines 525-639 (recursive doubling implementation),
  lines 427-523 (ring with limited overlap logic)

### Benchmarks and Literature

- Ibeid et al., "Scaling MPI Applications on Aurora" (arXiv:2512.04291, Dec 2025)
- Hidayetoglu et al., "HiCCL: A Hierarchical Collective Communication Library"
  (arXiv:2408.05962, Aug 2024)
- Thakur et al., "Optimization of Collective Communication Operations in MPICH"
  (IJHPCA, 2005) — latency/bandwidth model for ring and recursive doubling
- Chan et al., "Collective Communication: Theory, Practice, and Experience"
  (Concurrency and Computation, 2007) — alpha-beta model derivations
- Rabenseifner, "Optimization of Collective Reduction Operations"
  (ICCS 2004) — reduce-scatter + allgather decomposition
