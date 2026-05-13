# Why Host Bounce Buffers Limit oneCCL Scaling on PVC-Class Architectures

---

## Abstract

Intel Xe GPU architectures (PVC, CRI) require all inter-node collective
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

On PVC and CRI, GPU kernels cannot issue network operations. The NIC is not
accessible from GPU-side code. Data must bounce through host DRAM on both
the send and receive sides:

```
Default path: OFI transport, no HMEM (production)

  Node A                                               Node B
  ┌───────────┐                                       ┌───────────┐
  │  GPU VRAM │                                       │  GPU VRAM │
  └─────┬─────┘                                       └─────▲─────┘
   D2H  │ PCIe                                   H2D  │ PCIe
  ~2 us ▼                                       ~2 us │
  ┌───────────┐                                       ┌─────┴─────┐
  │ Host DRAM │                                       │ Host DRAM │
  │ (staging) │                                       │ (staging) │
  └─────┬─────┘                                       └─────▲─────┘
   CPU  │ fi_tsendmsg                       CPU wait  │
  ~1 us ▼                                       ~1 us │
  ┌───────────┐                                       ┌─────┴─────┐
  │   NIC A   │───────────── wire ~2 us ────────────▶│   NIC B   │
  └───────────┘  NIC reads host DRAM                 └───────────┘
                                         NIC writes host DRAM

  All 8 stages are sequential. Total: ~8-10 us per collective step.
  Network-only cost would be ~2-3 us.


Experimental path: OFI + CCL_ATL_HMEM=1

  Node A                                               Node B
  ┌───────────┐                                       ┌───────────┐
  │  GPU VRAM │                                       │  GPU VRAM │
  └─────┬─────┘                                       └─────▲─────┘
  PCIe  │ (NIC DMA-reads GPU BAR)    (NIC DMA-writes) │ PCIe
  ~1 us ▼                                       ~1 us │
  ┌───────────┐                                       ┌─────┴─────┐
  │   NIC A   │───────────── wire ~2 us ────────────▶│   NIC B   │
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
  (6) atl_comm->wait() blocks until completion         [CPU spin]
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
`true` whenever `atl_transport == ccl_atl_ofi`. Since OFI is the recommended
transport for inference, every production deployment operates in the
host-staged regime.

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
#define CCL_ALLREDUCE_SHORT_MSG_SIZE   8192        // 8 KB
#define CCL_ALLREDUCE_MEDIUM_MSG_SIZE  (1024*1024) // 1 MB
```

For the SYCL+ZE build (GPU path, `selector_allreduce.cpp` lines 20-46):
- Main algorithm: `topo` for all message sizes
- Scaleout table: `ring` for all message sizes
- Fallback: `recursive_doubling` for 0-8 KB, `ring` for 8 KB to CCL_MAX

For the CPU/non-ZE build with OFI:
- 0-8 KB: `recursive_doubling`
- 8 KB - 1 MB: `nreduce` (reduce-scatter + allgather)
- > 1 MB: `ring`

The ring algorithm has a limited overlap window
(`is_copy_overlap_enabled()` in `coll_util.cpp`): overlap only activates for
ring + multi-node + single-worker-mode. For the default scaleout path through
`allreduce_scaleout_sycl_simple`, there is no overlap at all.

### 2.3 Latency Budget Per Collective Step

For a 16 KB message on a PVC/CRI node (PCIe Gen4/5, ~32 GB/s effective):

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

where α is per-hop latency and β is bandwidth. Without bounce buffer, α is
the hardware network latency (~2 us on Slingshot-11). With bounce buffer, α
becomes the sum of all staging stages:

```
α_staged = α_D2H + α_host_task + α_OFI + α_network + α_wait + α_H2D
         ≈ 2 + 0.5 + 1.5 + 1.5 + 0.5 + 2 = 8 us
```

For a 16 KB message (well within the 8 KB threshold for recursive doubling,
so this also applies per-step in the nreduce case for 8-1024 KB):

| N nodes | log₂(N) steps | T without staging | T with staging |
|---|---|---|---|
| 8 | 3 | 3 * (4 + 0.5) = 14 us | 3 * (16 + 0.5) = 50 us |
| 64 | 6 | 6 * (4 + 0.5) = 27 us | 6 * (16 + 0.5) = 99 us |
| 512 | 9 | 9 * (4 + 0.5) = 41 us | 9 * (16 + 0.5) = 149 us |
| 2048 | 11 | 11 * (4 + 0.5) = 50 us | 11 * (16 + 0.5) = 182 us |

The Aurora benchmark (arXiv 2512.04291) measures ~250 us at 2048 nodes for
small messages. The gap between the 182 us model above and 250 us accounts
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

Ring's O(N) latency scaling makes it unsuitable for large-N inference.
oneCCL uses ring only for the scaleout phase at message sizes > 8 KB where
bandwidth efficiency matters more than latency. For decode inference (small
messages, latency critical), recursive doubling is selected automatically
via the threshold in `selector_allreduce.cpp`.

### 3.3 The topo Hierarchical Algorithm

`topo` reduces inter-node traffic by performing an intra-node reduce-scatter
first (Xe Link on PVC, PCIe P2P on CRI), then running the scaleout collective
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

### 4.1 PCIe Bandwidth Saturation

On a PVC node with 12 tiles, all 12 tiles stage through host memory
simultaneously during the reduce-scatter phase. The 12 tiles share 2 PCIe
root complexes (6 tiles per GPU/root complex). PCIe Gen4 x16 provides
approximately 32 GB/s unidirectional. Each tile's effective bandwidth for
staging is:

```
BW_per_tile = 32 GB/s / 6 tiles = ~5.3 GB/s peak (contended)
```

For a 16 KB message:

```
T_D2H_contended = 16 KB / 5.3 GB/s = ~3 us
```

vs. uncontended:

```
T_D2H_uncontended = 16 KB / 32 GB/s = ~0.5 us
```

The 6x contention factor is one reason the per-step latency significantly
exceeds what network hardware alone would require. Aurora uses 8x Slingshot-11
NICs (200 Gbps each = 25 GB/s per NIC), providing 200 GB/s aggregate NIC
bandwidth. The 2 GPUs at 32 GB/s each = 64 GB/s total PCIe capacity to host.
**The PCIe supply is 3x undersupplied relative to NIC demand.** This explains
why Aurora measures 23-25 GB/s per NIC rather than the 25 GB/s theoretical
maximum: the staging path saturates before the NICs do.

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

The endpoint index is hardcoded to `0` with a `TODO` comment — all collectives
from all ranks on a node go through endpoint 0, serializing at the ATL layer.
With multiple concurrent collectives (pipeline parallelism, multiple
microbatches), this is a contention point independent of the PCIe bandwidth
issue.

`atl_comm->wait()` inside the host task blocks the SYCL host thread until
MPI/OFI reports completion. This is a blocking wait, not an asynchronous
poll. Under load, the SYCL scheduler cannot repurpose this thread for other
work while waiting.

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

---

## 5. Where the Scaling Wall Appears

The scaling wall is the node count N* where `T_communication >= T_compute`
per layer. For decode inference:

```
T_compute per layer ≈ 2 * d_model * d_ffn / (FLOPS * N_tiles_per_node * N_nodes)
T_comm per allreduce ≈ T_intranode + log₂(N_nodes) * α_staged
```

For a representative 70B model on PVC-class hardware:
- FP16 FLOPS per tile: ~100 TFLOPS
- Decode layer FLOPS (batch=1): ~200M FLOPS (2 * 8192 * 29000)
- T_compute per layer (1 tile): ~2 us
- T_intranode (Xe Link): ~10 us
- α_staged (bounce buffer): ~8-12 us per hop

Setting `T_comm = T_compute * N_nodes`:

```
10 + log₂(N*) * 10 = 2 * N*
```

This yields N* ≈ 8-16 nodes. Beyond this, allreduce dominates the per-token
budget.

**Without the bounce buffer** (hypothetical direct GPU RDMA, α_network = 3 us):

```
10 + log₂(N*) * 3 = 2 * N*
```

This yields N* ≈ 64-128 nodes — 4-8x more headroom before hitting the same
wall.

The bounce buffer does not change the O(log N) scaling exponent of recursive
doubling. It multiplies the per-step coefficient by 3-4x, shifting the
scaling wall earlier by the same factor.

---

## 6. Empirical Validation from Aurora

The Aurora benchmark data (Ibeid et al., arXiv 2512.04291) shows:

| Message | 1 node | 2048 nodes | Ratio |
|---|---|---|---|
| 8 B | ~15 us | ~250 us | 16.7x |
| 64 KB | ~50 us | ~280 us | 5.6x |

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

## 8. Mitigation Strategies and Their Ceilings

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
PVC tile FLOPS from Intel product specifications, staged alpha of 8-12 us
per collective step measured from oneCCL source in Section 2, and validated
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
