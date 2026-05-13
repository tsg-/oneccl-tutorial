# oneCCL Overview

## Overview

**oneAPI Collective Communications Library (oneCCL)** is Intel's distributed communication
library for deep learning workloads. It provides a unified API across CPUs and Intel GPUs (XPU),
integrates with PyTorch via `torch.distributed`, and maps directly onto Intel MPI / libfabric
transports at the bottom.

oneCCL's design goals differ from NCCL in ways that matter for inference:
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
   
   For GPU buffers with no override → selects `topo`.

3. **Schedule construction**: The selected algorithm builds a **schedule** — a directed
   acyclic graph of **entries** (atomic operations). For `topo` allreduce:
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
- `scatter` / `gather` -- one-to-many / many-to-one (P0 in CRI roadmap)

---

## Where oneCCL Lives in the Inference Stack

For a TP=4 decode cluster on CRI hardware:

```
 Rank 0    Rank 1    Rank 2    Rank 3
  GPU0      GPU1      GPU2      GPU3
   │         │         │         │
   └─────────┴──CCL────┴─────────┘
           NUMA / PCIe fabric
           (no XeLink on CRI)
```

Each GPU holds 1/4 of the model weight. After every linear layer, an `allreduce` is needed
to sum the partial activations. That allreduce crosses NUMA via PCIe -- topology awareness
in ring construction is critical. See [Topology and Algorithm Selection](02_topology).

The critical-path latency budget for TP decode is tight. For a target TPOT of 50ms with
80 transformer layers and 2 allreduces per layer (attention output + FFN down projection):

```
160 allreduces × latency_per_allreduce ≤ 5ms comm budget (10% of TPOT)
→ latency_per_allreduce ≤ 31 µs

At batch=1, hidden=8192: message = 16 KB (BF16)
This is deep in the latency-bound regime — startup cost dominates.
```

---

## Key Environment Variables

| Variable | Purpose | Recommended for Inference |
|---|---|---|
| `CCL_ATL_TRANSPORT` | `mpi` or `ofi` | `ofi` (lower overhead) |
| `CCL_WORKER_COUNT` | oneCCL worker threads per process | `1` (Intel recommends ≤ 1 for GPU buffers) |
| `CCL_LOG_LEVEL` | `error`, `warn`, `info`, `debug` | `warn` in prod |
| `CCL_ALLREDUCE` | Main allreduce algorithm. Default `topo` for GPU builds. **Setting any value other than `topo` forces GPU data through a host-staged CPU path** — never set this for GPU inference. Control only the scaleout phase via `CCL_ALLREDUCE_SCALEOUT` instead. | Leave unset for GPU buffers (see [Perf Tuning](perf_tuning)) |
| `CCL_PRIORITY` | Task priority mode | `lifo` for low-latency |
| `I_MPI_PIN_DOMAIN` | MPI rank-to-core pinning | Match NUMA domains |
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

oneCCL cannot assume GPU fabric (CRI has none) and cannot fuse collectives into GPU kernels
(Xe compute EUs cannot initiate network I/O). Instead, oneCCL's `topo` algorithm:
- Uses **copy engines** for intra-node P2P (Level Zero IPC handles)
- Uses **host-staged OFI transport** for inter-node communication
- Relies on **overlap with compute** (via `async_op`) to hide latency

This architectural difference means oneCCL's optimization surface is fundamentally different:
NCCL optimizes kernel fusion and NVLink scheduling; oneCCL optimizes copy engine utilization,
host staging pipeline depth, and NUMA-aware ring construction.

For Intel XPU inference: **oneCCL is the only production-grade option.**
