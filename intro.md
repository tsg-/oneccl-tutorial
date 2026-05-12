# oneCCL for Inference Engineers

This tutorial covers Intel's **oneAPI Collective Communications Library (oneCCL)** from the perspective
of inference infrastructure -- specifically Tensor Parallelism (TP), Mixture-of-Experts (MoE) expert
routing, and disaggregated Prefill/Decode (PD) architectures.

:::{note}
This is an internal Intel AI Group / CCO Team resource. Hardware context throughout assumes
**CRI nodes: NUMA-only topology, no UALink/XeLink fabric.** Algorithm selection guidance is
calibrated accordingly.
:::

## Who This Is For

**If you are new to distributed ML or collective communication:** start with
[From First Principles: Collective Communication](chapters/00_foundations). It explains what
ranks, allreduce, and ring algorithms are from scratch, with math and ASCII diagrams. Nothing
in the rest of the tutorial assumes prior knowledge beyond that chapter.

**If you know MPI/NCCL basics already:** you can start at
[oneCCL Overview](chapters/01_overview) and use the Foundations chapter as a reference.

## What You Will Learn

- What collective communication is and why it exists (ranks, allreduce, ring algorithm)
- Which algorithm families work on NUMA-only hardware and why
- How oneCCL fits into the oneAPI stack and where it hands off to NIXL
- How to initialize oneCCL, run core collectives, and measure performance
- End-to-end TP decode loop with real oneCCL calls

## Prerequisites

- Python 3.9+, PyTorch 2.1+
- Intel MPI or IMPI installed (`module load intel/mpi`)
- `oneccl_bindings_for_pytorch` installed (see [Environment Setup](notebooks/02_environment_setup))

No prior knowledge of distributed training, MPI, or collective communication is assumed.
The [Foundations chapter](chapters/00_foundations) covers all necessary background.

## Quick Reference: CRI Collective Priorities

| Collective | Primary Algorithm | Status | Inference Use Case |
|---|---|---|---|
| Allreduce | Ring | Done | TP layer boundary sync |
| Allreduce | One-Shot | P1 | Low-latency decode (small batch, GPU fabric required) |
| Allgather | Ring | Done | Sequence parallelism |
| Alltoall | topo (scale-up + scatter scaleout) | Done | MoE expert routing |
| Scatter/Gather | Ring | P0 | KV cache distribution |

**Status:** Done = implemented and available on CRI today. P0 = highest priority,
actively in development. P1 = planned, pending hardware support (e.g., GPU fabric).

## How to Run the Notebooks

Each notebook is designed to be launched via `mpirun` from a JupyterHub terminal:

```bash
module load intel/mpi
mpirun -n 4 python -m ipykernel ...
```

For convenience, every notebook includes a **launcher cell** that writes and executes the MPI
workload as a subprocess, so you can run it from a single-rank kernel and still see multi-rank output.

## Architecture Boundary: oneCCL vs NIXL

```
┌─────────────────────────────────────────────────────────┐
│                   Inference Workload                    │
├──────────────────────┬──────────────────────────────────┤
│     oneCCL (CRI)     │           NIXL / UCX             │
│                      │                                  │
│  TP Allreduce        │  KV cache P→D transfer           │
│  MoE Alltoall        │  Disaggregated routing           │
│  SP Allgather        │  Large directed P2P              │
│  (synchronization)   │  (data movement)                 │
└──────────────────────┴──────────────────────────────────┘
```

oneCCL owns **synchronization primitives** across TP/EP ranks.
NIXL owns **KV cache data movement** between prefill and decode nodes.
These layers do not overlap. See [The oneCCL/NIXL Boundary](chapters/nixl_boundary) for details.
