# oneCCL for Inference Engineers

This tutorial covers Intel's **oneAPI Collective Communications Library (oneCCL)** from the perspective
of inference infrastructure -- specifically Tensor Parallelism (TP), Mixture-of-Experts (MoE) expert
routing, and disaggregated Prefill/Decode (PD) architectures.

## Audience

**If you are new to distributed ML or collective communication:** start with
[From First Principles: Collective Communication](chapters/00_foundations). It explains what
ranks, allreduce, and ring algorithms are from scratch, with math and ASCII diagrams. Nothing
in the rest of the tutorial assumes prior knowledge beyond that chapter.

**If you know MPI/NCCL basics already:** you can start at
[oneCCL Overview](chapters/01_overview) and use the Foundations chapter as a reference.

**If you are a training engineer (distributed training, ZeRO, pipeline parallelism):** the
Foundations and Topology chapters apply directly — the same algorithms run for training.
The key difference is message size regime: training gradient allreduces and ZeRO communication
are bandwidth-bound where inference decode is latency-bound. Jump to
[When to Use Which Collective §Training](chapters/03_when_to_use) for the training-specific
decision tree and ZeRO pattern guide.

## How to Navigate This Tutorial

Use this page as the map. The tutorial has three layers:

| If you want to... | Start here | Then read |
|---|---|---|
| Learn the concepts from first principles | [Foundations](chapters/00_foundations) | [oneCCL Overview](chapters/01_overview), [Topology](chapters/02_topology) |
| Decide which collective belongs in a workload | [When to Use Which Collective](chapters/03_when_to_use) | [The oneCCL / NIXL Boundary](chapters/nixl_boundary) |
| Run the examples | [Environment Setup](notebooks/02_environment_setup) | [Allreduce](notebooks/03a_allreduce_walkthrough), [Allgather](notebooks/03b_allgather), [Alltoall](notebooks/03c_alltoall_moe), then [TP Decode](notebooks/04_inference_tp_decode) |
| Tune or debug a deployment | [Performance Tuning Reference](chapters/perf_tuning) | [Debugging and Profiling](chapters/debugging_profiling) |
| Understand host-staged scaleout on PVC/Aurora | [PVC: Host-Staged Scaleout](chapters/pvc_scalability) | [Host Staging: The Scaling Wall](chapters/host_staging_scaling) |
| Integrate oneCCL from C++ / SYCL | [Native C++ / SYCL Integration](chapters/native_cpp_api) | [Performance Tuning Reference](chapters/perf_tuning) |

The main path is:

```text
Foundations -> oneCCL Overview -> Topology -> Hardware Constraints
			-> When to Use Which Collective -> Environment Setup
			-> Allreduce -> Allgather -> Alltoall -> End-to-End TP Decode
```

The reference chapters are intentionally more detailed. Read them when you need a specific
answer about tuning, profiling, native C++ integration, or PVC/Aurora scaleout behavior.

## Scope

- What collective communication is and why it exists (ranks, allreduce, ring algorithm)
- Which algorithm families work on NUMA-only hardware and why
- Training collective patterns: DP gradient sync, ZeRO stages 1/2/3, pipeline parallelism P2P
- How oneCCL fits into the oneAPI stack and where it hands off to NIXL (inference only)
- How to initialize oneCCL, run core collectives, and measure performance
- End-to-end TP decode loop with real oneCCL calls

## Prerequisites

- Python 3.9+, PyTorch 2.1+
- Intel MPI or IMPI installed (`module load intel/mpi`)
- `oneccl_bindings_for_pytorch` installed (see [Environment Setup](notebooks/02_environment_setup))

No prior knowledge of distributed training, MPI, or collective communication is assumed.
The [Foundations chapter](chapters/00_foundations) covers all necessary background.

## Quick Reference: BMG/CRI Collective Priorities

**Inference:**

| Collective | Primary Algorithm | Status | Inference Use Case |
|---|---|---|---|
| Allreduce | Ring | Done | TP layer boundary sync |
| Allreduce | One-Shot | P1 | Low-latency decode (small batch, GPU fabric required) |
| Allgather | Ring | Done | Sequence parallelism |
| Alltoall | topo (scale-up + scatter scaleout) | Done | MoE expert routing |
| Scatter/Gather | Ring | P0 | KV cache distribution |

**Training:**

| Collective | Primary Algorithm | Status | Training Use Case |
|---|---|---|---|
| Allreduce | Ring (BW-optimal) | Done | DP gradient sync (large gradients, BW-bound) |
| ReduceScatter | Ring | Done | ZeRO-2/3 gradient aggregation |
| Allgather | Ring | Done | ZeRO-3 parameter reconstruction before forward pass |
| Broadcast | Scatter+Allgather | Done | Checkpoint/weight broadcast at startup |
| P2P Send/Recv | (not a collective) | Done | Pipeline parallelism stage-to-stage activation handoff |

**Status:** Done = implemented and available on BMG/CRI today. P0 = highest priority,
actively in development. P1 = planned, pending hardware support (e.g., GPU fabric).

## Running the Notebooks

Notebooks are numbered to match the concepts they exercise: notebook `02` corresponds
to environment setup, `03a`–`03c` to the three core collectives, and `04` to the
end-to-end decode loop. There is no notebook `01` — the Overview chapter is concepts-only.
Start with `02_environment_setup` to verify your stack before running any benchmark.

Each notebook is designed to be launched via `mpirun` from a JupyterHub terminal:

```bash
module load intel/mpi
mpirun -n 4 python -m ipykernel ...
```

For convenience, every notebook includes a **launcher cell** that writes and executes the MPI
workload as a subprocess, so you can run it from a single-rank kernel and still see multi-rank output.

## Architecture Boundary: oneCCL vs NIXL

```
┌─────────────────────────────────────────────────┐
│                Inference Workload               │
├──────────────────────┬──────────────────────────┤
│  oneCCL (BMG/CRI)    │  NIXL / UCX              │
│                      │                          │
│  TP Allreduce        │  KV cache P→D transfer   │
│  MoE Alltoall        │  Disaggregated routing   │
│  SP Allgather        │  Large directed P2P      │
│  (synchronization)   │  (data movement)         │
└──────────────────────┴──────────────────────────┘
```

oneCCL owns **synchronization primitives** across TP/EP ranks.
NIXL owns **KV cache data movement** between prefill and decode nodes.
These layers do not overlap. See [The oneCCL/NIXL Boundary](chapters/nixl_boundary) for details.
