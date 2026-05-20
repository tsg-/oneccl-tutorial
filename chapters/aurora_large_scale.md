# Training and MoE at Scale on Aurora: Host Staging as a Bandwidth Ceiling

## Abstract

This chapter addresses the scaling regime that the decode-focused analysis in
[PVC: Scaleout Mechanisms](pvc_scalability) and
[Host-Staged Collectives §3-5](host_staging_scaling) does not cover: large-message
collectives at 1,000–10,000+ nodes on Aurora.

At this scale, the dominant question is not per-step latency (the decode
problem) but **aggregate throughput**: can the collective move data fast enough
to keep GPUs fed? The host staging mechanism is the same — every inter-node
byte transits host DRAM — but the symptom is different: NIC underutilization
rather than latency accumulation.

This chapter models Aurora's actual hardware parameters (8 NICs per node, 200
GB/s aggregate NIC bandwidth, 614 GB/s DDR5 host DRAM, 6 GPUs with PCIe Gen5
x16 each), derives the throughput ceiling imposed by host staging, and
identifies where oneCCL's multi-rail utilization limits effective bandwidth at
scale.

**Reader routing:**
- If your problem is decode latency at 8-64 nodes → [Host-Staged Collectives §3-5](host_staging_scaling)
- If your problem is training or MoE throughput at 1000+ nodes → this chapter

---

## 1. Aurora Node Architecture for Collective Communication

Each Aurora node contains:

```text
┌─────────────────────────────────────────────────────────────────────┐
│  Aurora Node                                                        │
│                                                                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│  │ GPU 0   │ │ GPU 1   │ │ GPU 2   │ │ GPU 3   │ │ GPU 4   │ │ GPU 5   │
│  │(2 tiles)│ │(2 tiles)│ │(2 tiles)│ │(2 tiles)│ │(2 tiles)│ │(2 tiles)│
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
│       │PCIe Gen5   │           │           │           │           │
│       │x16 (64GB/s)│           │           │           │           │
│  ═════╪════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════
│       │     Xe Link mesh (all-to-all, ~100+ GB/s per link)         │
│  ═════╪════════════╪═══════════╪═══════════╪═══════════╪═══════════╪═════
│       │            │           │           │           │           │
│  ┌────┴────────────┴───────────┴──┐  ┌────┴───────────┴───────────┴──┐
│  │     CPU Socket 0               │  │     CPU Socket 1               │
│  │  Xeon Max (HBM + DDR5-4800)    │  │  Xeon Max (HBM + DDR5-4800)   │
│  │  8-ch DDR5 ≈ 307 GB/s          │  │  8-ch DDR5 ≈ 307 GB/s         │
│  └──────────┬───────┬─────────────┘  └──────────┬───────┬────────────┘
│          PCIe│Gen4   │Gen4                    PCIe│Gen4   │Gen4
│          x16 │       │x16                     x16 │       │x16
│  ┌───────┐ ┌┴──────┐│┌───────┐ ┌───────┐  ┌───────┐┌┴──────┐┌───────┐┌───────┐
│  │NIC 0  │ │NIC 1  │││NIC 2  │ │NIC 3  │  │NIC 4  ││NIC 5  ││NIC 6  ││NIC 7  │
│  │200Gbps│ │200Gbps│││200Gbps│ │200Gbps│  │200Gbps││200Gbps││200Gbps││200Gbps│
│  └───────┘ └───────┘│└───────┘ └───────┘  └───────┘└───────┘└───────┘└───────┘
│                      │                                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Key bandwidth parameters (from Allcock et al., Goto et al., Ibeid et al.):**

| Path | Bandwidth | Notes |
|---|---|---|
| GPU ↔ CPU (PCIe Gen5 x16) | 64 GB/s per GPU | bidirectional |
| GPU ↔ GPU (Xe Link) | ~100+ GB/s per link | intra-node only |
| CPU DRAM (DDR5-4800, 8-ch) | ~307 GB/s per socket | shared by all GPUs on that socket |
| NIC (Slingshot-11, 200 Gbps) | 25 GB/s per NIC | 8 NICs per node |
| NIC ↔ CPU (PCIe Gen4 x16) | 32 GB/s per NIC | the NIC-facing PCIe is Gen4, not Gen5 |
| Aggregate NIC per node | 200 GB/s | theoretical; measured ~23 GB/s per NIC (Ibeid et al.) |

The critical observation: **GPU data reaches NICs only through host DRAM.**
There is no direct GPU-to-NIC path in the default configuration. Data flows:

```text
GPU VRAM → PCIe Gen5 → CPU DRAM → PCIe Gen4 → NIC → wire
```

Each inter-node byte traverses two PCIe hops and one DRAM write+read.

---

## 2. The Throughput Ceiling from Host Staging

For large-message collectives (training gradients, MoE dispatch), bandwidth
matters more than per-step latency. The question is: what is the maximum
sustained collective throughput per node?

### 2.1 Per-Node Bandwidth Budget

The data path has three potential bottlenecks:

```text
GPU → CPU DRAM:    6 GPUs × 64 GB/s = 384 GB/s aggregate (PCIe Gen5)
CPU DRAM itself:   2 sockets × 307 GB/s = 614 GB/s (but shared with compute)
CPU DRAM → NICs:   8 NICs × 32 GB/s = 256 GB/s aggregate (PCIe Gen4)
NICs → wire:       8 NICs × 25 GB/s = 200 GB/s aggregate (link rate)
```

The narrowest point in the staging pipeline is the **NIC aggregate**: 200 GB/s.
But the measured effective throughput is lower. Ibeid et al. report ~23 GB/s
per NIC for GPU-memory buffers (vs 25 GB/s theoretical), giving ~184 GB/s
aggregate. The gap comes from:

- PCIe Gen4↔Gen5 conversion overhead at the NIC-facing ports
- CPU-side OFI posting overhead (fi_tsendmsg + progress)
- DDR5 read contention when all 8 NICs DMA from host DRAM simultaneously

### 2.2 Effective Collective Throughput

The `allreduce_scaleout_sycl_simple` path handles the entire scaleout message
as a single chunk (no internal pipelining — `TODO: chunking/pipelining`). The
timeline for one scaleout collective is:

```text
T_scaleout = T_D2H + T_MPI_internal + T_H2D

Where:
  T_D2H  = message_size / PCIe_BW      (GPU VRAM → host staging buffer)
  T_MPI  = MPI_Allreduce on host buf   (MPI can pipeline internally)
  T_H2D  = message_size / PCIe_BW      (host staging buffer → GPU VRAM)
```

The D2H and H2D are strict bookends: the NIC is idle during both. MPI's
internal algorithm may achieve near-wire-rate during its phase (Intel MPI
pipelines ring/tree steps internally), but the entire D2H must complete before
MPI starts, and the entire MPI phase must complete before H2D starts.

The NIC utilization over the full collective is:

```text
NIC utilization = T_MPI / (T_D2H + T_MPI + T_H2D)
```

For a concrete example — 14 MB message (the per-node gradient share at 10k
nodes for a 70B model, after `topo` intra-node reduce-scatter produces 1/12
of the layer gradient):

```text
T_D2H = 14 MB / 64 GB/s ≈ 0.21 ms
T_MPI = tree allreduce on 14 MB across 10k nodes ≈ variable (see §4.3)
T_H2D = 14 MB / 64 GB/s ≈ 0.21 ms

D2H + H2D overhead = 0.42 ms per collective, regardless of network time
```

If T_MPI is 1 ms (realistic for a 14-step tree at scale):
```text
NIC utilization = 1.0 / (0.21 + 1.0 + 0.21) ≈ 70%
```

If T_MPI is 0.5 ms (smaller message or fewer steps):
```text
NIC utilization = 0.5 / (0.21 + 0.5 + 0.21) ≈ 54%
```

The staging tax is the D2H + H2D time that adds to every collective. For
training gradients, this is ~0.4 ms per layer's allreduce — a fixed overhead
that HMEM would eliminate entirely.

### 2.3 Staging Buffer Size Limit

The staging buffer is pre-allocated at communicator init. If the scaleout
message exceeds it, oneCCL falls back entirely:

```cpp
if (comm->get_scaleout_host_buf_size() < count * ccl_dtype.size()) {
    LOG_WARN("Falling back. TODO: chunking/pipelining");
    done = false;
    return e;
}
```

For training workloads, the scaleout message is the post-reduce-scatter chunk:
layer_gradient_size / tiles_per_node. For a 70B model's largest layer (~1.6 GB)
with 12 tiles per node, the scaleout message is ~137 MB. Whether this exceeds
the default staging buffer depends on the build configuration — if it does,
oneCCL falls back from the SYCL scaleout path to the scheduler-based fallback,
which uses ring allreduce with its own (separate) staging mechanism.

In practice, large-scale training frameworks (DeepSpeed, FSDP) issue per-bucket
allreduces of ~25 MB each, well within typical staging buffer limits. The risk
is with unusually large single-layer gradients or non-standard bucket sizes.

---

## 3. Multi-NIC Utilization in oneCCL

### 3.1 Multi-Rail Support in ATL

Aurora's 8 NICs appear as 8 separate OFI endpoints (or MPI rails). Whether
oneCCL utilizes all 8 depends on the transport layer:

**MPI transport (`CCL_ATL_TRANSPORT=mpi`):** Intel MPI's multi-rail support
(`I_MPI_FABRICS=shm:ofi`) can stripe large messages across multiple NICs
automatically. MPI's internal algorithm (ring, recursive doubling, etc.)
distributes traffic across available rails based on message size and the
configured rail policy. The oneCCL `host_task` mechanism simply delegates
to `MPI_Allreduce`, leaving multi-rail striping entirely to the MPI layer.

**OFI transport (`CCL_ATL_TRANSPORT=ofi`):** oneCCL's ATL/OFI layer creates
endpoints against the libfabric provider. While some code paths post to
a specific ATL endpoint index, any advanced multi-rail utilization in the
direct OFI path would require explicit endpoint management and
message splitting at the oneCCL level.

**The Host Task Bottleneck:** Regardless of how many rails are engaged, the scaleout
path submits collective calls into a `sycl::host_task`. This delegates networking
execution back to the host CPU, heavily loading the SYCL runtime scheduler. The
dependency queues and linear searches in the SYCL scheduler ultimately bottleneck
system progress well before rail striping can be fully exploited.

### 3.2 Implications at 1000+ Nodes

With 12 ranks per node and 8 NICs, the per-node effective bandwidth depends
on how traffic is distributed across NICs:

```text
MPI multi-rail (ATL/MPI, large messages):
  MPI internally stripes each rank's allreduce across available rails.
  All 8 NICs active → per-node BW ≈ 184 GB/s.
  This is the expected case on Aurora for large-scale training jobs.

Static NIC assignment (ATL/OFI, or MPI without multi-rail):
  Each rank is bound to one NIC based on NUMA placement.
  12 ranks across 8 NICs → some NICs serve 2 ranks (contended).
  Per-node BW still ≈ 184 GB/s (all NICs active), but individual
  ranks on a shared NIC see 11.5 GB/s each vs 23 GB/s for unshared.
  Load imbalance causes the slowest rank to gate the collective.
```

With MPI transport (the default on Aurora for large-scale jobs), multi-rail is
handled inside Intel MPI. The effective per-node throughput is determined by
MPI's multi-rail efficiency, not oneCCL's endpoint index. The diagnostic in
§6.1 confirms whether all NICs are carrying traffic.

---

## 4. Training Gradient Allreduce at 1000+ Nodes

### 4.1 Data Volume

For a 70B model (BF16 parameters), ring allreduce moves:

```text
Data per rank per step = 2 × (p-1)/p × 140 GB ≈ 280 GB  (for large p)
```

At 10,000 nodes (120,000 tiles), each rank's share is tiny — but the ring has
120,000 steps and each step moves M/p = 140 GB / 120,000 ≈ 1.2 MB per rank.

### 4.2 Ring Allreduce Time

```text
T_ring = 2(p-1) × α + 2 × M / BW_effective

For p = 120,000 (12 tiles/node × 10,000 nodes):
  Latency term: 2 × 120,000 × α_staged ≈ 240,000 × 8 µs = 1.92 seconds
  Bandwidth term: 2 × 140 GB / BW_effective_per_rank
```

The latency term alone (1.9 seconds for ring at 120k ranks with 8 µs per step)
shows why **ring allreduce is not used at this scale.** oneCCL and MPI both
switch to tree-based algorithms (recursive doubling, recursive halving-doubling,
or SMP-aware trees) that have O(log N) step count.

### 4.3 Tree Allreduce at 10,000 Nodes

With a tree algorithm (reduce-scatter + allgather, each with log₂(N) steps),
the per-step data volume shrinks at each level (reduce-scatter halves data per
step). But the host staging overhead applies to the total scaleout message, not
per-step. The relevant structure is:

```text
T_tree_scaleout = T_D2H(M_scaleout) + T_MPI_tree(M_scaleout, N_nodes) + T_H2D(M_scaleout)
```

where `M_scaleout` is the message entering the scaleout phase (after intra-node
reduce-scatter via Xe Link shrinks it by 12×).

For a 70B model's gradient allreduce with `topo`:
- Full gradient per layer: ~1.6 GB (largest layer)
- After intra-node reduce-scatter (12 tiles): 1.6 GB / 12 ≈ 137 MB
- With training frameworks bucketing to ~25 MB: M_scaleout ≈ 25 MB

At 10,000 nodes:

```text
T_D2H = 25 MB / 64 GB/s ≈ 0.39 ms
T_MPI = MPI_Allreduce(25 MB, 10k nodes) — MPI internally uses tree/recursive halving
      ≈ log₂(10,000) × (α_network + M_step / BW_per_partner)
      ≈ 14 × (2 µs + 25 MB / 100 GB/s)
      ≈ 14 × (2 µs + 250 µs) ≈ 3.5 ms
T_H2D = 25 MB / 64 GB/s ≈ 0.39 ms

Total = 0.39 + 3.5 + 0.39 = 4.3 ms per bucket
```

(BW_per_partner ≈ 100 GB/s assumes MPI stripes across ~4 NICs to each tree
partner. This is an estimate — actual throughput depends on dragonfly routing
and MPI's multi-rail policy. The key structural point is that D2H + H2D adds
a fixed 0.78 ms regardless of the MPI-internal time.)

Without staging (HMEM — NIC DMAs from GPU directly):
```text
T_MPI ≈ 3.5 ms (same — MPI still runs its tree algorithm)
No D2H/H2D bookends.
Total = 3.5 ms per bucket
```

The staging overhead per bucket is **0.78 ms** (the D2H + H2D bookends). For
a full 70B gradient allreduce with ~6 buckets in flight, the overhead is
additive only if buckets are serialized. With gradient bucketing pipelined
against backward compute, partial overlap is possible — but the bookends
still serialize within each bucket's scaleout phase.

### 4.4 Where the Ceiling Bites

The host staging overhead matters when gradient allreduce time exceeds the
backward compute time available to overlap with it.

For a 70B model at batch=32, seq_len=2048, on 10,000 nodes:

```text
FLOPs per step ≈ 2 × 70×10⁹ × 32 × 2048 = 9.2 × 10¹⁵ FLOPs
FLOPs per node (12 tiles, ~100 TFLOPS BF16 per tile) = 1.2 PFLOPS
T_compute = 9.2 × 10¹⁵ / (10,000 × 1.2 × 10¹⁵) = 0.77 ms

T_comm per bucket (staged) = 4.3 ms
T_comm per bucket (HMEM)   = 3.5 ms
```

At 10,000 nodes with batch=32: **communication exceeds compute regardless of
staging**, but staging adds 0.78 ms per bucket (the D2H + H2D bookends).

The fix at this scale is not removing staging (though that helps) — it's
**increasing batch size** to grow T_compute while T_comm stays constant:

```text
batch=128: T_compute = 3.1 ms → comm fraction = 4.3 / (3.1 + 4.3) = 58%
batch=512: T_compute = 12.3 ms → comm fraction = 4.3 / (12.3 + 4.3) = 26%
```

With HMEM (eliminating staging overhead):
```text
batch=128: T_compute = 3.1 ms, T_comm = 3.5 ms → comm fraction = 53%
batch=512: T_compute = 12.3 ms, T_comm = 3.5 ms → comm fraction = 22%
```

**The staging overhead at 10k nodes adds ~0.8 ms per gradient bucket** — the
D2H + H2D bookends that HMEM would eliminate. Across 6 serialized buckets per
step, that's up to ~5 ms of staging overhead per training iteration. At
batch=512, staging costs ~4 percentage points of efficiency (26% vs 22%).
The dominant cost at 10k nodes is MPI's tree allreduce time itself, not the
staging bookends — but the bookends are the only part under oneCCL's control.

---

## 5. MoE All-to-All at 1000+ Nodes

### 5.1 Traffic Pattern at Scale

MoE expert parallelism distributes experts across nodes. At 1000+ nodes, the
all-to-all dispatch pattern becomes:

```text
N_nodes = 1000, 12 tiles/node, 256 experts
Experts per node = 256 / 1000 ≈ 0.256  (some nodes hold 1 expert, most hold 0)
```

This means: with more nodes than experts, expert parallelism stops scaling.
Practical MoE deployments at 1000+ nodes combine EP with DP:

```text
Realistic: EP across 32 nodes (1 node per 8 experts), DP across remaining
  EP communicator size = 32 nodes × 12 tiles = 384 ranks
  DP communicator size = 1000 / 32 = ~31 replicas
```

The all-to-all within the EP group is the bottleneck.

### 5.2 Per-Node Staging Load

With 32 nodes in the EP communicator:

Each tile dispatches tokens to every EP node. For batch=128, hidden=7168,
top-k=8, 32 EP nodes:

```text
Dispatch per tile = batch × top_k × hidden × 2 bytes / 32
                  = 128 × 8 × 7168 × 2 / 32 ≈ 459 KB per destination node

Total outbound per tile = 31 × 459 KB ≈ 14.3 MB
Total outbound per node (12 tiles) = 12 × 14.3 MB ≈ 171 MB
```

All 12 tiles stage simultaneously through host DRAM for the dispatch phase:

```text
D2H aggregate = 171 MB, all tiles concurrent
DDR5 available = 614 GB/s (dual-socket)
T_D2H = 171 MB / 614 GB/s ≈ 0.28 ms

NIC outbound = 171 MB to wire
NIC available = 184 GB/s effective
T_NIC = 171 MB / 184 GB/s ≈ 0.93 ms

T_H2D ≈ T_D2H ≈ 0.28 ms (receiving side)
```

Total dispatch all-to-all per MoE layer:
```text
T_dispatch ≈ 0.28 + 0.93 + 0.28 ≈ 1.49 ms (staged)
T_dispatch ≈ 0.93 ms (HMEM, NIC-limited only)
```

With combine (same traffic, reverse direction): total MoE communication per
layer ≈ 2 × 1.49 = **2.98 ms staged**, or 2 × 0.93 = **1.86 ms with HMEM**.

### 5.3 MoE Scaling as EP Group Grows

As the EP communicator grows (more nodes per EP group), each tile sends to more
destinations but less data per destination:

| EP nodes | Experts/node | Data per tile outbound | NIC time | Staging overhead |
|---|---|---|---|---|
| 8 | 32 | 7 × 1.8 MB = 12.6 MB | 0.82 ms | 0.34 ms |
| 32 | 8 | 31 × 459 KB = 14.3 MB | 0.93 ms | 0.56 ms |
| 64 | 4 | 63 × 229 KB = 14.4 MB | 0.94 ms | 0.56 ms |
| 128 | 2 | 127 × 115 KB = 14.6 MB | 0.95 ms | 0.56 ms |

Total outbound volume is approximately constant (each tile sends the same
total amount regardless of fan-out). The staging overhead grows slightly
because more smaller messages have higher per-message OFI posting overhead.
The NIC time is roughly constant.

**The MoE scaling constraint at 1000+ nodes is not the all-to-all communication
volume** (which stays constant per tile). It's:

1. **Number of concurrent OFI posts:** 31-127 destinations require 31-127
   separate fi_tsendmsg calls per tile. Each goes through host staging
   sequentially unless the MPI/OFI layer batches them internally.

2. **Small-message OFI overhead:** At 128 EP nodes, each message is 115 KB —
   small enough that per-message posting overhead (~5 µs) dominates transfer
   time (115 KB / 23 GB/s = 5 µs). The collective becomes latency-bound rather
   than bandwidth-bound.

3. **Incast at destination:** Each node receives from 31-127 sources
   simultaneously. The receiving H2D path must handle 171 MB of inbound data
   scattered across 12 tiles, all arriving within ~1 ms.

### 5.4 Practical Consequence

For DeepSeek-scale MoE on Aurora at 32-128 EP nodes:
- MoE communication adds ~3 ms per layer (2 all-to-alls, staged)
- Expert compute per layer ≈ 1-5 ms (depends on expert size, batch)
- Communication fraction: 37-75%

Host staging adds ~1 ms per layer above the NIC-limited floor. This is less
dramatic than the decode latency case (where staging is 3-4× the network
cost) but accumulates across 60+ MoE layers in DeepSeek-class models:

```text
Total MoE comm overhead from staging ≈ 60 layers × 1 ms = 60 ms per step
```

At batch=128, a training step on DeepSeek-R1 at 32 EP nodes takes ~200-400 ms.
The 60 ms staging tax is 15-30% of step time — meaningful but not dominant.
At larger EP groups (64-128 nodes), the per-message OFI overhead grows and
staging becomes a larger fraction.

---

## 6. Diagnosing Multi-Rail and Staging at Scale

### 6.1 Confirming Multi-NIC Utilization

```bash
# Check which NICs are active per rank
fi_info -p cxi -l 2>/dev/null | grep "domain:"
# Should show cxi0, cxi1, ... cxi7 (one per NIC)

# Check MPI's rail assignment
export I_MPI_DEBUG=5
mpirun -n 24 -ppn 12 hostname 2>&1 | grep "rail"

# Monitor NIC utilization during a training step
# (requires access to Slingshot counters or node monitoring)
watch -n 0.1 cat /sys/class/cxi/cxi*/device/telemetry/tx_bytes
```

### 6.2 Confirming Staging Path

```bash
export CCL_LOG_LEVEL=info
# Look for:
#   "use_hmem: 0" → staging active (the default)
#   "use_hmem: 1" → HMEM active (NIC DMA from GPU BAR)

# For SYCL path specifically:
export CCL_LOG_LEVEL=debug
# Look for:
#   "copy_to_host: 1" → host staging in SYCL scaleout path
#   "Falling back. TODO: chunking/pipelining" → message exceeded staging buffer
```

### 6.3 Measuring Effective Bandwidth

```python
import torch
import torch.distributed as dist
import time

dist.init_process_group('ccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

# Measure large-message allreduce bandwidth
for size_mb in [1, 10, 100, 1000]:
    tensor = torch.randn(size_mb * 1024 * 1024 // 4, dtype=torch.float32,
                         device='xpu')
    # Warmup
    for _ in range(5):
        dist.all_reduce(tensor)
    torch.xpu.synchronize()

    start = time.perf_counter()
    for _ in range(20):
        dist.all_reduce(tensor)
    torch.xpu.synchronize()
    elapsed = (time.perf_counter() - start) / 20

    algbw = (size_mb / 1024) / elapsed  # GB/s
    busbw = algbw * 2 * (world_size - 1) / world_size  # bus bandwidth
    if rank == 0:
        print(f"{size_mb:6d} MB  algbw={algbw:.1f} GB/s  busbw={busbw:.1f} GB/s")
```

Expected results on Aurora with host staging:
- At 2 nodes: busbw ≈ 20-40 GB/s (limited by single-rail staging path)
- At 64 nodes: busbw ≈ 80-150 GB/s (MPI multi-rail engaged for large messages)
- At 1000+ nodes: busbw ≈ 100-184 GB/s (tree algorithm, multi-rail)

If busbw is significantly below 100 GB/s at 1000+ nodes, the bottleneck is
likely multi-rail underutilization or staging pipeline inefficiency — not
network capacity.

---

## 7. Tuning Guidance for 1000+ Node Training

### 7.1 Configuration for Large-Scale Training on Aurora

```bash
# Transport
export CCL_ATL_TRANSPORT=mpi      # Use MPI for multi-rail support
export I_MPI_FABRICS=shm:ofi      # Shared memory intra-node, OFI inter-node

# Multi-rail (Intel MPI)
export I_MPI_OFI_PROVIDER=cxi     # Slingshot provider
# MPI auto-detects multi-rail on cxi; explicit control if needed:
# export I_MPI_MULTIRAIL=1

# NUMA binding
export I_MPI_PIN_DOMAIN=socket    # Each rank pinned to its GPU's socket
export I_MPI_PIN_ORDER=compact    # Keep ranks on same socket adjacent

# oneCCL algorithm — leave unset for training at 1000+ nodes
# MPI internally selects tree-based algorithms at this scale.
# CCL_ALLREDUCE_SCALEOUT=ring is only useful for <64 node jobs with large messages.
# Do NOT set ring at 1000+ nodes — ring's O(N) latency term dominates (see §4.2).

# Worker threads
export CCL_WORKER_COUNT=2         # 2 workers for high collective rate at scale

# HMEM (if verified active — see §6.2)
export CCL_ATL_HMEM=1
```

### 7.2 When to Use Which Allreduce Algorithm at Scale

| Regime | Message size | Node count | Algorithm | Why |
|---|---|---|---|---|
| Decode TP | 16 KB | 8-64 | Leave default (recursive doubling via MPI) | Latency-bound, minimize steps |
| Prefill TP | 1-64 MB | 8-64 | `CCL_ALLREDUCE_SCALEOUT=ring` | Bandwidth-bound at low node count; ring's O(N) latency is fine at N≤64 |
| Training DP | 14+ GB total, sharded | 100-10,000 | Leave default (MPI selects tree) | Tree has O(log N) latency + ring-like BW |
| Training ZeRO-2 | reduce-scatter | 100-10,000 | Leave default | MPI's internal selection is appropriate |
| MoE alltoall | 100 KB - 10 MB per dest | 32-128 (EP) | No scaleout override available | oneCCL uses MPI_Alltoallv directly |

### 7.3 Batch Size as the Primary Scaling Lever

At 1000+ nodes, the most effective way to maintain compute efficiency is
increasing batch size. The communication cost for tree allreduce is roughly
constant per step (it depends on model size, not batch size). Compute scales
linearly with batch:

```text
Efficiency(batch) = T_compute(batch) / (T_compute(batch) + T_comm)
                  = (batch × C) / (batch × C + T_comm)
```

For 70B model at 10,000 nodes (T_comm ≈ 4.3 ms per bucket staged, assume
gradient allreduce overlaps partially with backward compute — effective
non-overlapped comm ≈ last bucket on critical path):

```
batch=32:   T_compute = 0.77 ms → efficiency = 15%  (comm-dominated)
batch=128:  T_compute = 3.1 ms  → efficiency = 42%
batch=512:  T_compute = 12.3 ms → efficiency = 74%
batch=2048: T_compute = 49 ms   → efficiency = 92%
```

The host staging tax of ~0.8 ms per bucket (the D2H + H2D bookends) shifts the
efficiency curve. At batch=512 with HMEM (T_comm = 3.5 ms): efficiency would
be ~78% vs 74% without — modest at this batch size. The staging tax matters
more at smaller batch sizes where it's a larger fraction of T_comm.

### 7.4 Gradient Accumulation as an Alternative

If memory limits prevent large batches, gradient accumulation achieves the
same effect: accumulate N micro-batches of gradients locally before one
allreduce. The communication cost is amortized across N steps:

```text
T_effective_comm = T_comm / N_accumulation_steps
```

At 10,000 nodes with 4× gradient accumulation:
```text
T_effective_comm = 4.3 ms / 4 = 1.08 ms per micro-step
batch=32 effective efficiency: 0.77 / (0.77 + 1.08) = 42% (vs 15% without accumulation)
```

---

## 8. What Changes with HMEM at Scale

If `CCL_ATL_HMEM=1` is confirmed active (see [PVC: Scaleout Mechanisms §7](pvc_scalability)):

| Metric | Staged (current) | HMEM (if active) | Improvement |
|---|---|---|---|
| NIC utilization during collective | ~80% (D2H/H2D bookends) | ~95% (NIC DMA direct) | 1.2× |
| Per-bucket gradient allreduce (25 MB, 10k nodes) | ~4.3 ms | ~3.5 ms | 1.2× |
| Staging overhead per bucket | 0.78 ms | 0 ms | eliminated |
| MoE dispatch (32 EP nodes) | ~1.5 ms | ~0.9 ms | 1.7× |
| Training efficiency at batch=512 | 74% | 78% | modest |

HMEM eliminates the D2H and H2D bookends. The MPI-internal phase is unchanged
(MPI still runs on host buffers unless it too integrates HMEM). The remaining
overhead after HMEM is CPU-posted `fi_tsendmsg` (the CPU still initiates each
transfer) and NIC DMA latency over PCIe Gen4.

**At 10k-node scale, HMEM matters more for MoE than for training gradients.**
For gradient allreduce, the MPI-internal tree time (~3.5 ms) dominates and the
bookends (0.78 ms) are ~18% of total collective time. For MoE dispatch, the
bookends (0.56 ms) are a larger fraction of the shorter NIC-limited phase
(0.93 ms), making HMEM's relative impact stronger (1.7× vs 1.2×).

For decode (§5 of host_staging_scaling), HMEM saves ~4 µs per step × 14
steps = 56 µs per allreduce — modest in absolute terms but critical because
each µs is on the token-latency critical path.

---

## 9. Summary

At 1000-10,000 nodes on Aurora, oneCCL's host staging mechanism adds a
**fixed per-collective overhead** (D2H + H2D bookends) on top of the MPI-
internal allreduce time. This is qualitatively different from the decode
regime (where staging dominates the per-step cost):

1. **The D2H and H2D bookends** add ~0.8 ms per 25 MB gradient bucket — time
   during which the NICs are idle. At this scale, MPI's internal tree allreduce
   takes ~3.5 ms, so the bookends are ~18% of total collective time (not the
   dominant cost).

2. **Multi-NIC utilization depends on MPI's multi-rail layer**, not oneCCL
   directly. The SYCL scaleout path uses `host_task`s which delegate to
   `MPI_Allreduce`, which can internally stripe across rails. However, the
   SYCL scheduler overhead restricts throughput before multi-rail limits
   are strictly breached.

3. **For training at this scale**, the primary lever is batch size, not staging
   removal. At batch=512, communication fraction is 26% (staged) vs 22%
   (HMEM) — a small gap. At batch=128, the gap is wider (58% vs 53%). The
   staging tax is most painful when batch size is memory-constrained.

4. **For MoE at this scale**, staging overhead is proportionally larger because
   the all-to-all traffic pattern has shorter NIC time relative to bookend time.
   HMEM reduces MoE dispatch latency by 1.7× (vs 1.2× for training allreduce).

5. **Network topology effects** (dragonfly adaptive routing, inter-group
   congestion, job placement) are outside oneCCL's control and outside this
   analysis. At 10k nodes, MPI's tree allreduce time may itself be elevated
   by network congestion — measure NIC utilization to determine whether the
   bottleneck is staging, network, or both.

---

## References

### Aurora Hardware

- Allcock et al., "Aurora: Architecting Argonne's First Exascale Supercomputer
  for Accelerated Scientific Discovery" (arXiv:2509.08207) — node architecture,
  ECB topology, PCIe Gen5 x16 GPU→CPU, 8 Slingshot NICs per node
- Goto et al., "Sustaining Exascale Performance: Lessons from HPL and HPL-MxP
  on Aurora" (arXiv:2604.09517) — PCIe Gen4 NIC-facing ports, memory bandwidth
- Ibeid et al., "Scaling MPI Applications on Aurora" (arXiv:2512.04291) —
  measured NIC bandwidth (~23 GB/s per NIC for GPU buffers), allreduce scaling

### oneCCL Source

- `src/coll/algorithms/allreduce/sycl/allreduce_scaleout_sycl.cpp` — staging
  pipeline, host_task scheduler loading, copy_to_host override
- `src/coll/coll_util.cpp` — enable_hmem gate, host buffer allocation

### Related Chapters

- [PVC: Scaleout Mechanisms](pvc_scalability) — HMEM verification (§7),
  source-level evidence for host staging
- [Host-Staged Collectives](host_staging_scaling) — decode latency model (§3-5),
  per-step overhead derivation (§2)
