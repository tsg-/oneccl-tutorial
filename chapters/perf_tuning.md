# Performance Tuning Reference

This chapter is a reference card for every `CCL_*` and `I_MPI_*` environment variable
that matters for inference on CRI hardware. All settings assume NUMA-only topology (no
XeLink/UALink).

---

## How oneCCL Reads Configuration

oneCCL reads environment variables at `dist.init_process_group()` time. Variables set after
init have no effect. Set them before launching MPI:

```bash
# Correct: set before mpirun
CCL_ATL_TRANSPORT=ofi CCL_WORKER_COUNT=1 mpirun -n 4 python script.py

# Also correct: export in shell before mpirun
export CCL_ATL_TRANSPORT=ofi
export CCL_WORKER_COUNT=1
mpirun -n 4 python script.py

# Wrong: set inside Python after init_process_group
os.environ["CCL_WORKER_COUNT"] = "2"   # too late — has no effect
dist.init_process_group(backend="ccl") # already read env
```

---

## oneCCL Variables

### Transport Layer

| Variable | Values | Default | Recommendation |
|---|---|---|---|
| `CCL_ATL_TRANSPORT` | `ofi`, `mpi` | `mpi` | **`ofi`** — lower overhead, no PMI roundtrip per collective |

`ofi` uses libfabric directly. `mpi` goes through Intel MPI's collective layer, adding
PMI synchronization overhead. For inference (latency-critical), `ofi` is strictly better
once the MPI environment is initialized.

```bash
export CCL_ATL_TRANSPORT=ofi
```

---

### Worker Threads

| Variable | Values | Default | Recommendation |
|---|---|---|---|
| `CCL_WORKER_COUNT` | Integer ≥ 1 | `1` | `1` for decode, `2` for prefill |
| `CCL_WORKER_AFFINITY` | Core list, e.g. `2,3` | Auto | Pin to non-NUMA-boundary cores |

Worker threads are the oneCCL internal threads that drive the collective progress engine.
More workers can help throughput-bound prefill workloads. For decode (latency-bound, one
collective at a time), extra workers just waste CPU cores and add scheduling noise.

```bash
# Decode: single worker, let MPI rank own the core
export CCL_WORKER_COUNT=1

# Prefill: two workers can help overlap large allreduces
export CCL_WORKER_COUNT=2
export CCL_WORKER_AFFINITY=4,5   # pin to specific cores away from GPU NUMA domain
```

---

### Algorithm Selection

| Variable | Values | Default | Recommendation |
|---|---|---|---|
| `CCL_ALLREDUCE` | `ring`, `recursive_doubling`, `rabenseifner`, `nreduce`, `direct`, `auto` | `auto` | **`ring`** on NUMA-only |
| `CCL_ALLGATHER` | `ring`, `flat`, `multi_bcast`, `auto` | `auto` | `ring` or `auto` |
| `CCL_ALLTOALL` | `scatter_gather`, `scatter_gather_barrier`, `topo`, `auto` | `auto` | `auto` (uses hierarchical) |
| `CCL_REDUCE_SCATTER` | `ring`, `direct`, `auto` | `auto` | `ring` or `auto` |
| `CCL_BCAST` | `ring`, `double_tree`, `naive`, `auto` | `auto` | leave `auto` |

**When to override vs. leave auto:**
- Override `CCL_ALLREDUCE=ring` if you observe high p95 variance — `auto` can occasionally
  select a suboptimal algorithm for edge-case message sizes
- Leave `CCL_ALLTOALL` on `auto` — the hierarchical algorithm selection is non-trivial and
  the auto path handles it correctly
- For benchmarking, always pin the algorithm to isolate its performance

> **Why ring for allreduce on NUMA-only:** See [Foundations §4.5](00_foundations) and
> [Topology & Algorithm Selection](02_topology). One-Shot and Direct algorithms require
> simultaneous fan-out which serializes on shared PCIe. Ring pipelines through the fabric.

```bash
export CCL_ALLREDUCE=ring
export CCL_ALLGATHER=ring
```

---

### Logging and Diagnostics

| Variable | Values | Default | Recommendation |
|---|---|---|---|
| `CCL_LOG_LEVEL` | `error`, `warn`, `info`, `debug`, `trace` | `warn` | `warn` in production, `info` to verify ring construction |
| `CCL_ITT_LEVEL` | `0`, `1`, `2` | `0` | `1` for VTune profiling |

```bash
# Verify ring construction — look for "ring order" in output
CCL_LOG_LEVEL=info mpirun -n 4 python script.py 2>&1 | grep -i "ring"

# In production
export CCL_LOG_LEVEL=warn
```

---

### Priority and Scheduling

| Variable | Values | Default | Recommendation |
|---|---|---|---|
| `CCL_PRIORITY` | `none`, `lifo` | `none` | `lifo` for decode (last-in-first-out prioritizes the newest collective) |
| `CCL_SPIN_COUNT` | Integer | Platform default | Increase to reduce yield latency for small messages |

`CCL_PRIORITY=lifo` tells the progress engine to deprioritize stale collective handles.
This matters if you have async collectives pending from a previous iteration that ran long.

---

### Memory and Buffer Tuning

| Variable | Values | Default | Recommendation |
|---|---|---|---|
| `CCL_CHUNK_COUNT` | Integer | Auto | For ring algorithm: number of pipeline stages. Increasing can improve bandwidth utilization for very large messages. Leave auto for inference. |
| `CCL_MIN_CHUNK_SIZE` | Bytes | Auto | Minimum chunk size in ring pipeline. Auto is usually correct. |
| `CCL_BUFFER_SIZE` | Bytes | Auto | Internal staging buffer. May need increase for very large alltoall. |
| `CCL_ZE_COPY_ENGINE` | `main`, `link` | `main` | On systems with copy engines, `link` uses the dedicated copy engine |

---

## Intel MPI Variables

These affect how MPI ranks are launched and pinned. Wrong pinning breaks topology-aware ring
construction.

### Rank-to-Core Pinning

| Variable | Values | Recommendation |
|---|---|---|
| `I_MPI_PIN_DOMAIN` | `auto`, `socket`, `numa`, `core`, `node` | **`socket`** for 2-socket nodes (ranks 0-N/2 on socket 0, rest on socket 1) |
| `I_MPI_PIN_ORDER` | `scatter`, `compact` | `compact` (keep ranks on same socket adjacent) |
| `I_MPI_PIN_CELL` | `core`, `unit` | `core` |

```bash
# 8 ranks across 2 sockets (4 GPUs per socket):
I_MPI_PIN_DOMAIN=socket \
I_MPI_PIN_ORDER=compact \
mpirun -n 8 -ppn 8 python script.py
```

**Verify pinning is correct:**
```bash
I_MPI_DEBUG=4 mpirun -n 4 python -c "import os; print(os.getenv('PMI_RANK'), os.getenv('NUMA_NODE'))" 2>&1 | grep "pinned"
```

---

### Transport and Fabric

| Variable | Values | Recommendation |
|---|---|---|
| `I_MPI_FABRICS` | `shm:ofi`, `shm:mxm`, `ofi`, `tcp` | **`shm:ofi`** — shared memory for intra-node, OFI for inter-node |
| `I_MPI_OFI_PROVIDER` | `psm3`, `verbs`, `tcp` | `psm3` for Intel fabric, `verbs` for Mellanox/ConnectX |
| `I_MPI_SHM` | `0`, `1` | `1` (enable shared memory for intra-node) |

```bash
export I_MPI_FABRICS=shm:ofi
export I_MPI_SHM=1
```

---

## Complete Production Launch Template

```bash
#!/bin/bash
# Decode inference: 8 GPUs, 2 sockets × 4 GPUs/socket, batch=1

# Transport
export CCL_ATL_TRANSPORT=ofi
export I_MPI_FABRICS=shm:ofi

# Algorithm
export CCL_ALLREDUCE=ring
export CCL_ALLGATHER=ring

# Workers: 1 per rank for decode
export CCL_WORKER_COUNT=1

# Priority: LIFO for decode hot path
export CCL_PRIORITY=lifo

# Logging: minimal in production
export CCL_LOG_LEVEL=warn

# Pinning: socket-level for topology-aware ring
export I_MPI_PIN_DOMAIN=socket
export I_MPI_PIN_ORDER=compact

mpirun -n 8 -ppn 8 python decode_server.py
```

---

## Complete Debug/Profiling Launch Template

```bash
#!/bin/bash
# Use this when diagnosing performance issues

export CCL_ATL_TRANSPORT=ofi
export CCL_ALLREDUCE=ring
export CCL_WORKER_COUNT=1
export I_MPI_PIN_DOMAIN=socket
export I_MPI_FABRICS=shm:ofi

# Enable ring construction logging
export CCL_LOG_LEVEL=info

# Enable VTune ITT markers (record with `vtune -collect hotspots`)
export CCL_ITT_LEVEL=1

mpirun -n 8 -ppn 8 python debug_script.py 2>&1 | tee /tmp/ccl_debug.log

# After run: check that ring order is topology-aware
grep -i "ring order\|topo\|rank order" /tmp/ccl_debug.log
```

---

## Troubleshooting Common Issues

### Collective hangs / deadlock

```bash
# 1. Ensure all ranks call the collective — even ranks that "have nothing to do"
#    must participate; collective is a barrier for all ranks

# 2. Check for mismatched sizes
CCL_LOG_LEVEL=debug mpirun -n 4 python script.py 2>&1 | grep "count\|mismatch"

# 3. Check MPI initialized correctly
mpirun -n 4 python -c "
import torch.distributed as dist
import oneccl_bindings_for_pytorch
dist.init_process_group(backend='ccl')
print(f'rank {dist.get_rank()} of {dist.get_world_size()} initialized OK')
dist.destroy_process_group()
"
```

### High p95 variance (jitter)

```bash
# 1. Verify NUMA pinning — mismatched ranks cause extra cross-socket hops
numactl --hardware
I_MPI_DEBUG=4 mpirun -n 4 python -c "import time; time.sleep(1)" 2>&1 | grep "pinned to"

# 2. Check for background processes competing on PCIe
cat /proc/interrupts | grep -i "nvme\|eth\|ib" | head -10

# 3. Try increasing spin count to avoid yield latency
export CCL_SPIN_COUNT=10000
```

### Alltoallv correctness errors

```bash
# Alltoallv requires send and receive counts to be consistent across all ranks
# Each rank's recv_counts[j] must equal rank j's send_counts[my_rank]
# If they don't match, you get silently wrong data or a hang

# Verify with a simple round-trip test:
python verify_alltoallv.py  # see notebooks/03c_alltoall_moe for a reference test
```
