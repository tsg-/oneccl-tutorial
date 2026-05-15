# oneCCL Overview

## Overview

**oneAPI Collective Communications Library (oneCCL)** is Intel's distributed communication
library for deep learning workloads. It provides a unified API across CPUs and Intel GPUs (XPU),
integrates with PyTorch via `torch.distributed`, and maps directly onto Intel MPI / libfabric
transports at the bottom.

oneCCL's design goals differ from NCCL in ways that matter for distributed ML:
- **Multi-transport**: supports both MPI and OFI (libfabric) at the transport layer, allowing
  deployment on Intel fabrics (OPA, Slingshot) without vendor lock-in
- **CPU + GPU unified API**: the same collective call works for host tensors and device tensors;
  the library internally selects the appropriate path
- **Topology awareness**: the `topo` algorithm constructs hierarchical schedules based on
  detected Level Zero device topology, NUMA distances, and fabric connectivity

---

## Internal Architecture

```
┌─────────────────────────────────────────────────────────┐
│              PyTorch / Framework Layer                  │
├─────────────────────────────────────────────────────────┤
│        torch.distributed (CCL backend)                  │  ← user API
├─────────────────────────────────────────────────────────┤
│          oneccl_bindings_for_pytorch                    │  ← Python bindings
├─────────────────────────────────────────────────────────┤
│                  oneCCL Core                            │
│   ┌─────────────────────────────────────────────────┐   │
│   │  Collective Selector (algorithm dispatch)       │   │
│   │  Schedule Builder (DAG of entries)              │   │
│   │  Progress Engine (worker threads)               │   │
│   └─────────────────────────────────────────────────┘   │
├──────────────────────┬──────────────────────────────────┤
│   ATL/MPI Transport  │  ATL/OFI Transport (libfabric)   │
│   (Intel MPI)        │  (PSM3 / verbs / tcp)            │
├──────────────────────┼──────────────────────────────────┤
│   Level Zero (GPU)   │  Shared Memory (intra-node)      │
├──────────────────────┴──────────────────────────────────┤
│          Hardware: CPU / XPU / NIC / PCIe               │
└─────────────────────────────────────────────────────────┘
```

### Dispatch Path: `dist.all_reduce()` → Hardware

When your code calls `dist.all_reduce(tensor, op=ReduceOp.SUM)`:

1. **PyTorch dispatch**: `torch.distributed` routes to the registered CCL backend via
   `ProcessGroupCCL::allreduce()`. The binding checks tensor device, dtype, and count.

2. **Algorithm selection**: oneCCL's `selector_allreduce` examines:
   - Buffer location (host vs device memory)
   - Message size (thresholds: SHORT < 8192 elements, MEDIUM < 1048576)
   - World size and topology (detected at init)
   - `CCL_ALLREDUCE` environment override (if set)
   
   For GPU buffers on a SYCL+ZE build with no override → selects `topo`.

3. **Schedule construction**: The selected algorithm builds a **schedule** — a directed
   acyclic graph of **entries** (atomic operations). Conceptual example for a multi-node
   `topo` allreduce (actual entries depend on topology and message size):
   ```
   Entry 1: reduce_scatter via Level Zero IPC (intra-node, scale-up)
   Entry 2: copy result to host staging buffer (PCIe DMA)
   Entry 3: allreduce via OFI (inter-node, scaleout)
   Entry 4: copy result back to device (PCIe DMA)
   Entry 5: allgather via Level Zero IPC (intra-node, scale-up)
   ```

4. **Progress engine**: The CCL worker thread(s) execute schedule entries. Each entry
   may post Level Zero commands, OFI sends/receives, or local copies. The worker spins
   on completion handles until all entries complete.

5. **Synchronization**: For synchronous calls, `all_reduce()` blocks until the schedule
   completes. For `async_op=True`, a `Work` handle is returned; the user calls
   `work.wait()` later.

### Collective Selector Thresholds

The selector uses message count to choose between algorithm variants within a family.
From `selector_allreduce.cpp`:

| Range | Threshold | Algorithm Tendency |
|---|---|---|
| SHORT | < 8192 **elements** | Recursive doubling (minimize steps) |
| MEDIUM | 8192 – 1,048,576 **elements** | Ring or Rabenseifner |
| LONG | > 1,048,576 **elements** | Ring (maximize bandwidth) |

> **Note: thresholds are in element count, not bytes.** For BF16 (2 bytes/element):
> SHORT is < 16 KB; MEDIUM is 16 KB – 2 MB. For FP32 (4 bytes/element):
> SHORT is < 32 KB; MEDIUM is 32 KB – 4 MB.
> The 512 KB threshold cited in the Foundations chapter is the MPICH literature value
> from 2003 hardware; oneCCL's calibrated thresholds for Intel hardware are lower.

These thresholds are overridden when `topo` is selected (GPU path), since `topo` handles
its own internal decomposition.

> **Under the hood:** The selector logic is in `src/coll/selection/selector_allreduce.cpp`
> lines 20–46 (GPU/ZE path) and the threshold constants are in `src/coll/selection/selector.hpp`:
> `CCL_ALLREDUCE_SHORT_MSG_SIZE = 8192` (elements, not bytes). For BF16, that is 16 KB — the
> exact size of a TP decode allreduce at hidden=8192, batch=1. You can override these thresholds
> at runtime via `CCL_ALLREDUCE_SHORT_MSG_SIZE` environment variable if needed for non-standard
> hidden dimensions.

---

## Collective Operations Covered

oneCCL supports the full set of MPI-style collectives:

**Synchronization-heavy (inference critical):**
- `allreduce` -- sum/avg partial tensors across all ranks
- `allgather` / `allgatherv` -- each rank contributes a shard, all ranks get full tensor
- `alltoall` / `alltoallv` -- each rank sends distinct data to each other rank (MoE)
- `reduce_scatter` -- combine allreduce + scatter in one pass

**Point-to-collective:**
- `broadcast` -- one rank sends to all
- `reduce` -- all ranks contribute, result on one rank
- `scatter` / `gather` -- one-to-many / many-to-one (P0 in BMG/CRI roadmap)

---

## Where oneCCL Lives in the Stack

### Inference (Tensor Parallelism)

For a TP=4 decode cluster on BMG/CRI hardware:

```
 Rank 0    Rank 1    Rank 2    Rank 3
  GPU0      GPU1      GPU2      GPU3
   │         │         │         │
   └─────────┴──CCL────┴─────────┘
           NUMA / PCIe fabric
           (no XeLink on BMG/CRI)
```

Each GPU holds 1/4 of the model weight. After every linear layer, an `allreduce` is needed
to sum the partial activations. That allreduce crosses NUMA via PCIe — ring construction order matters here.
See [Topology and Algorithm Selection](02_topology).

The TP decode latency budget is tight. For a target TPOT of 50ms with
80 transformer layers and 2 allreduces per layer (attention output + FFN down projection):

```
160 allreduces × latency_per_allreduce ≤ 5ms comm budget (10% of TPOT)
→ latency_per_allreduce ≤ 31 µs

At batch=1, hidden=8192: message = 16 KB (BF16)
This is deep in the latency-bound regime — startup cost dominates.
```

### Training (Data-Parallel Gradient Sync)

The same topology applies for distributed training. Each GPU computes gradients on its
data shard; allreduce sums them before the optimizer step:

```
 Rank 0    Rank 1    Rank 2    Rank 3
  GPU0      GPU1      GPU2      GPU3
   │         │         │         │
   └─────────┴──CCL────┴─────────┘
     gradient allreduce (per layer)
```

The scalability question for training is bandwidth sufficiency, not per-collective
latency. For a 7B model (14 GB gradients, BF16), using ring allreduce at 25 GB/s
effective PCIe P2P bandwidth:

```
Ring data moved per rank = 2 × ((p-1)/p) × 14 GB ≈ 21 GB  (for p=4)
Effective time per allreduce = 21 GB / 25 GB/s ≈ 840 ms

This is amortized across layers — ZeRO-2/3 partition gradients per-layer so
each allreduce is ~layer_size/total_params × 840 ms, pipelined with compute.
The wall appears when gradient traffic saturates the PCIe/NIC bandwidth budget
before compute hides it.
```

For training: use `CCL_ALLREDUCE_SCALEOUT=ring` (the default). Ring is usually
the right choice at training message sizes (GB-scale, bandwidth-bound). The 512 KB crossover threshold from
[Foundations §7](00_foundations) is irrelevant for training — gradient messages are
3–4 orders of magnitude larger.

---

## Key Environment Variables

| Variable | Purpose | Recommended |
|---|---|---|
| `CCL_ATL_TRANSPORT` | `mpi` or `ofi` | `ofi` (lower overhead for both inference and training) |
| `CCL_WORKER_COUNT` | oneCCL worker threads per process | `1` for inference (latency-bound); `1–2` for training |
| `CCL_LOG_LEVEL` | `error`, `warn`, `info`, `debug` | `warn` in prod; `info` to diagnose algorithm selection |
| `CCL_ALLREDUCE` | Main allreduce algorithm. Default `topo` for SYCL+ZE GPU builds. **Setting a value other than `topo` can force fallback behavior and may stage GPU buffers through host memory.** For GPU workloads, control only the scaleout phase via `CCL_ALLREDUCE_SCALEOUT` instead. | Leave unset for GPU buffers (see [Perf Tuning](perf_tuning)) |
| `CCL_ALLREDUCE_SCALEOUT` | Algorithm for inter-node phase only. `ring` is bandwidth-optimal and correct for both inference and training. | `ring` |
| `CCL_PRIORITY` | Task priority mode | `lifo` for low-latency inference |
| `I_MPI_PIN_DOMAIN` | MPI rank-to-core pinning | `socket` — match NUMA domains for both inference and training |
| `I_MPI_FABRICS` | `shm:ofi`, `ofi`, `tcp` | `shm:ofi` |

See [Performance Tuning Reference](perf_tuning) for the full variable catalog.

---

## oneCCL vs Alternatives

| Library | Vendor | XPU Support | Inference Tuning | NUMA-Aware | Design Philosophy |
|---|---|---|---|---|---|
| **oneCCL** | Intel | Native | Growing | Yes (ring) | Multi-transport, topology-adaptive |
| **NCCL** | NVIDIA | No | Mature | NVLink-optimized | Kernel-fused, NVLink-first |
| **RCCL** | AMD | No | Moderate | ROCm fabric | NCCL port for MI-series |
| **UCC** | UCF/NVIDIA | Partial | Experimental | Depends | Modular, pluggable backends |
| **NIXL** | NVIDIA | Yes | KV transport only | Via UCX | Point-to-point data movement |

**Key differences from NCCL:**

NCCL is designed around a hardware assumption: high-bandwidth NVLink meshes within a node
(900 GB/s on H100) with InfiniBand GPUDirect RDMA between nodes. Its algorithms are
implemented as CUDA kernels that directly read/write peer GPU memory — the "kernel-fused"
approach means the collective operation runs entirely on the GPU with no host involvement.

oneCCL cannot assume GPU fabric (BMG/CRI has none) and cannot fuse collectives into GPU kernels
(Xe compute EUs cannot initiate network I/O). Instead, oneCCL's `topo` algorithm:
- Uses **copy engines** for intra-node P2P (Level Zero IPC handles)
- Uses **host-staged OFI transport** for inter-node communication
- Relies on **overlap with compute** (via `async_op`) to hide latency

The two libraries optimize different things: NCCL optimizes kernel fusion and NVLink
scheduling; oneCCL optimizes copy engine utilization, host staging pipeline depth, and
NUMA-aware ring construction.

For Intel XPU inference: **oneCCL is the primary supported collective path.**

**Next:** [Topology and Algorithm Selection](02_topology) — why ring construction order
matters on NUMA hardware and how oneCCL builds topology-aware rings automatically.
