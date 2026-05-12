# oneCCL Overview

## What Is oneCCL

**oneAPI Collective Communications Library (oneCCL)** is Intel's distributed communication
library for deep learning workloads. It provides a unified API across CPUs and Intel GPUs (XPU),
integrates with PyTorch via `torch.distributed`, and maps directly onto Intel MPI / libfabric
transports at the bottom.

```
┌─────────────────────────────────────────┐
│           PyTorch / Framework           │
├─────────────────────────────────────────┤
│     torch.distributed (CCL backend)     │  ← your code lives here
├─────────────────────────────────────────┤
│              oneCCL API                 │
├──────────────────┬──────────────────────┤
│   ATL/MPI        │   ATL/OFI            │  ← transport selection
│   (Intel MPI)    │   (libfabric)        │
├──────────────────┴──────────────────────┤
│      Hardware: CPU / XPU / NIC          │
└─────────────────────────────────────────┘
```

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
in ring construction is critical. See [Topology & Algorithm Selection](02_topology).

## Key Environment Variables

| Variable | Purpose | Recommended for Inference |
|---|---|---|
| `CCL_ATL_TRANSPORT` | `mpi` or `ofi` | `ofi` (lower overhead) |
| `CCL_WORKER_COUNT` | oneCCL worker threads per process | `1` (Intel recommends ≤ 1 for GPU buffers) |
| `CCL_LOG_LEVEL` | `error`, `warn`, `info`, `debug` | `warn` in prod |
| `CCL_ALLREDUCE` | Scale-up algorithm (default `topo` for GPU) | Leave unset for GPU buffers (see [Perf Tuning](perf_tuning)) |
| `CCL_PRIORITY` | Task priority mode | `lifo` for low-latency |
| `I_MPI_PIN_DOMAIN` | MPI rank-to-core pinning | Match NUMA domains |
| `I_MPI_FABRICS` | `shm:ofi`, `ofi`, `tcp` | `shm:ofi` |

## oneCCL vs Alternatives

| Library | Vendor | XPU Support | Inference Tuning | NUMA-Aware |
|---|---|---|---|---|
| **oneCCL** | Intel | Native | Growing | Yes (ring) |
| **NCCL** | NVIDIA | No | Mature | NVLink-optimized |
| **RCCL** | AMD | No | Moderate | ROCm fabric |
| **UCC** | UCF/NVIDIA | Partial | Experimental | Depends |
| **NIXL** | NVIDIA | Yes | KV transport only | Via UCX |

For Intel XPU inference: **oneCCL is the only production-grade option.**
