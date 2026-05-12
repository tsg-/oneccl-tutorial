# Collective Communication from First Principles

This chapter builds everything from scratch — why collectives exist, how the algorithms
work mathematically, and what drives algorithm selection on real hardware. The formulas
and thresholds in §4 and §6 come directly from the primary literature, not approximations.

---

## 1. Why Multiple Processes Need to Talk

A single GPU runs a program sequentially. When you scale a neural network across multiple
GPUs, each GPU runs a **separate process** with its own memory. They cannot read each
other's memory directly.

The problem: many deep learning operations require every GPU to have data from every
other GPU before the computation can continue.

Consider Tensor Parallelism (TP) in a transformer decoder:

```
GPU 0 computes: partial_output_0 = input @ weight_shard_0   (it holds 1/4 of W)
GPU 1 computes: partial_output_1 = input @ weight_shard_1
GPU 2 computes: partial_output_2 = input @ weight_shard_2
GPU 3 computes: partial_output_3 = input @ weight_shard_3

Goal: every GPU needs (partial_output_0 + partial_output_1 + partial_output_2 + partial_output_3)
```

You could do this naively: GPU 0 sends its partial to GPUs 1, 2, 3; GPU 1 sends its partial
to GPUs 0, 2, 3; etc. That is N×(N-1) point-to-point messages — quadratic growth, and every
link carries redundant traffic.

**Collective communication** solves these "all need all" problems efficiently by exploiting
the structure of what everyone is computing.

---

## 2. Ranks, World Size, and Communicators

Before any collective can run, the library needs a shared directory of who exists.

| Term | Meaning |
|---|---|
| **Rank** | Integer ID of a process within a group. Rank 0 is usually the "root." |
| **World size** (p) | Total number of processes in the group. |
| **Communicator** | A named group of ranks that can collectively communicate. |

In oneCCL via PyTorch:

```python
import torch.distributed as dist
import oneccl_bindings_for_pytorch  # registers the 'ccl' backend

dist.init_process_group(backend="ccl")   # reads rank/size from MPI env

rank       = dist.get_rank()             # my process ID: 0, 1, 2, ...
world_size = dist.get_world_size()       # p = total number of processes
```

> **Source:** MPI Forum. *MPI: A Message-Passing Interface Standard, Version 4.1*, §6:
> Groups, Contexts, Communicators, and Caching. 2023.

---

## 3. The Core Collectives — What They Do

There are four collectives that matter for inference. Everything else is a specialization.

### 3.1 Allreduce

**Every rank contributes a tensor. The reduced result (e.g. sum) lands on every rank.**

```
Before (p=4):
  Rank 0: [1, 2, 3]
  Rank 1: [4, 5, 6]
  Rank 2: [7, 8, 9]
  Rank 3: [10, 11, 12]

After allreduce(op=SUM):
  Rank 0: [22, 26, 30]   ← 1+4+7+10,  2+5+8+11,  3+6+9+12
  Rank 1: [22, 26, 30]
  Rank 2: [22, 26, 30]
  Rank 3: [22, 26, 30]
```

**Inference use case:** TP layer boundary. After a row-parallel linear layer, each GPU holds
a partial activation sum. Allreduce combines them so every GPU has the full activation.

### 3.2 Allgather

**Each rank contributes a distinct chunk. All ranks receive all chunks concatenated.**

```
Before:
  Rank 0: [A]       (my shard)
  Rank 1: [B]
  Rank 2: [C]
  Rank 3: [D]

After allgather:
  Rank 0: [A, B, C, D]
  Rank 1: [A, B, C, D]
  Rank 2: [A, B, C, D]
  Rank 3: [A, B, C, D]
```

**Inference use case:** Sequence Parallelism. Each rank holds a shard of the KV cache.
Before attention, all shards are gathered so every rank can attend over the full context.

### 3.3 Alltoall (and Alltoallv)

**Each rank sends a distinct chunk to every other rank. It is a distributed transpose.**

```
Before (rank i holds [data_for_r0, data_for_r1, data_for_r2, data_for_r3]):
  Rank 0: [A0, A1, A2, A3]
  Rank 1: [B0, B1, B2, B3]
  Rank 2: [C0, C1, C2, C3]
  Rank 3: [D0, D1, D2, D3]

After alltoall (rank i receives column i from everyone):
  Rank 0: [A0, B0, C0, D0]
  Rank 1: [A1, B1, C1, D1]
  Rank 2: [A2, B2, C2, D2]
  Rank 3: [A3, B3, C3, D3]
```

Alltoallv is the variable-count variant — each rank can send/receive different amounts to
each peer. This is required for MoE routing where token-to-expert assignment is imbalanced.

**Inference use case:** MoE expert dispatch — tokens routed to their selected expert GPU,
then expert outputs routed back to their origin GPUs.

### 3.4 Reduce-Scatter

**Like Allreduce but the result is split: rank i receives only the i-th chunk of the
reduced tensor.**

```
Before:
  Rank 0: [A0, B0, C0, D0]
  Rank 1: [A1, B1, C1, D1]
  Rank 2: [A2, B2, C2, D2]
  Rank 3: [A3, B3, C3, D3]

After reduce_scatter(op=SUM):
  Rank 0: [A0+A1+A2+A3]     ← only chunk 0
  Rank 1: [B0+B1+B2+B3]     ← only chunk 1
  Rank 2: [C0+C1+C2+C3]
  Rank 3: [D0+D1+D2+D3]
```

Allreduce = Reduce-Scatter followed by Allgather. If you only need one chunk of the result,
stop after Reduce-Scatter — you avoid the Allgather's bandwidth cost.

---

## 4. The α-β Cost Model

Before analyzing algorithms, you need the standard model for how long a message takes.

> "We assume that the time taken to send a message between any two nodes can be modeled
> as **α + nβ**, where α is the latency (or startup time) per message, independent of message
> size, β is the transfer time per byte, and n is the number of bytes transferred."
>
> — Thakur, R. & Gropp, W. (2003). *Improving the Performance of Collective Operations
> in MPICH.* PVM/MPI 2003, LNCS 2840, pp. 49–60. Argonne National Laboratory.

For reduction operations, a third term is added:

> "We assume that **γ** is the computation cost per byte for performing the reduction
> operation locally on any process."
>
> — ibid.

So the full model: **T = α + nβ + nγ** (for a combined communication + reduction step).

The network interface is assumed to be **single-ported**: at most one message sent and
one message received simultaneously. This is an important assumption — it means you
cannot exploit simultaneous multi-peer fan-out, which is exactly why One-Shot Allreduce
fails on PCIe-only hardware (see §5.5).

---

## 5. Algorithms — How Collectives Are Actually Implemented

The same collective can be implemented with many algorithms. The right choice depends
on message size, number of ranks, and hardware topology.

### 5.1 Naive Allreduce (Star / Centralized Reduce)

The obvious approach: all ranks send to rank 0, rank 0 reduces, rank 0 broadcasts.

```
Step 1 — Gather at root:
  Rank 1 ──→ Rank 0
  Rank 2 ──→ Rank 0
  Rank 3 ──→ Rank 0

Step 2 — Broadcast from root:
  Rank 0 ──→ Rank 1
  Rank 0 ──→ Rank 2
  Rank 0 ──→ Rank 3
```

Using the α-β model:
- Gather cost: (p-1)(α + nβ) + nγ
- Broadcast cost: (p-1)(α + nβ)
- **Total: 2(p-1)(α + nβ) + nγ**

Rank 0 handles 2(p-1) messages while every other rank handles 2. It is a serial bottleneck
that does not scale.

### 5.2 Allgather: Ring vs. Recursive Doubling

This is the pair analyzed in depth by Thakur & Gropp (2003). Both algorithms are optimal
in bandwidth; they differ only in latency.

**Ring allgather** (the classic algorithm):

> "In the first step, each process i sends its contribution to process i+1 and receives
> the contribution from process i−1 (with wrap-around). From the second step onwards
> each process i forwards to process i+1 the data it received from process i−1 in the
> previous step. If p is the number of processes, the entire algorithm takes p−1 steps.
> If n is the total amount of data to be gathered on each process, then at every step
> each process sends and receives n/p amount of data."
>
> — Thakur & Gropp (2003), §4.1

```
T_ring = (p-1)α + ((p-1)/p)nβ
```

**Recursive doubling allgather** (new in MPICH):

> "In the first step, processes that are a distance 1 apart exchange their data. In the
> second step, processes that are a distance 2 apart exchange their own data as well as
> the data they received in the previous step. In the third step, processes that are a
> distance 4 apart exchange their own data as well as the data they received in the
> previous two steps. In this way, for a power-of-two number of processes, all processes
> get all the data in lg p steps."
>
> — ibid.

```
T_rec_dbl = lg(p)α + ((p-1)/p)nβ
```

**Both have the same bandwidth term** `((p-1)/p)nβ` — neither can do better, since
each process must receive n/p data from p-1 other processes. The difference is only
in the latency term: `(p-1)α` for Ring vs. `lg(p)α` for Recursive Doubling.

**Algorithm selection rule from MPICH:**
- Messages **< 512 KB**: Recursive Doubling (fewer steps = lower latency)
- Messages **≥ 512 KB**: Ring

**Why ring wins for large messages — the critical insight from the paper:**

> "For long messages (> 512 KB), however, we find that recursive doubling runs much
> slower than the ring algorithm... We believe this difference is because of the difference
> in the communication pattern of the two algorithms: **The ring algorithm has a
> nearest-neighbor communication pattern, whereas in recursive doubling, processes that
> are much farther apart communicate.** To confirm this hypothesis, we used the b_eff MPI
> benchmark, which measures the performance of about 48 different communication patterns,
> and found that, for long messages on both the Myrinet cluster and the IBM SP, **some
> communication patterns (particularly nearest neighbor) achieve more than twice the
> bandwidth of other communication patterns.**"
>
> — Thakur & Gropp (2003), §4.1 (emphasis added)

This is not a theoretical result — it is a measured hardware effect. On real clusters,
nearest-neighbor transfers (adjacent ranks on a ring) achieve 2× or more bandwidth vs.
the long-distance pairs that recursive doubling requires. Ring's nearest-neighbor pattern
exploits physical locality in the switch fabric, NUMA topology, and cache coherence
domains. Recursive doubling's random-distance pattern saturates different links at
different rates.

**For NUMA-only CRI hardware**, this effect is even more pronounced: nearest-neighbor
means intra-socket hops (fast PCIe), while recursive doubling forces cross-socket UPI
transfers on many steps.

### 5.3 Broadcast: Binary Tree vs. Van de Geijn

**Binary tree broadcast:**

```
T_tree = ⌈lg p⌉(α + nβ)
```

Good for short messages (lg p latency steps). For long messages, the bandwidth term
scales as `n·lg(p)·β` which is sub-optimal.

**Van de Geijn algorithm** (Scatter + Allgather):

> "In this algorithm, the message to be broadcast is first divided up and scattered among
> the processes, similar to an MPI_Scatter; the scattered data is then collected back to
> all processes, similar to an MPI_Allgather... For very long messages where we use the
> ring allgather, the time taken by the broadcast is:"
>
> — Thakur & Gropp (2003), §4.2

```
T_vandegeijn = (lg p + p − 1)α + 2((p-1)/p)nβ
```

This reduces the bandwidth term from `n·lg(p)·β` (binary tree) to `2·((p-1)/p)·nβ`
(Van de Geijn) — a reduction proportional to lg(p)/2. For p=8 that is a 1.5× improvement;
for p=64 it is a 3× improvement.

**MPICH selection threshold:** binary tree for **< 12 KB**, Van de Geijn for **≥ 12 KB**.

### 5.4 Reduce-Scatter: Recursive Halving vs. Pairwise Exchange

Reduce-scatter can be implemented several ways.

**Recursive halving** (for commutative operations, short messages):

> "In the first step, each process exchanges data with a process that is a distance p/2
> away: Each process sends the data needed by all processes in the other half, receives
> the data needed by all processes in its own half, and performs the reduction operation
> on the received data. The reduction can be done because the operation is commutative.
> In the second step, each process exchanges data with a process that is a distance p/4
> away..."
>
> — Thakur & Gropp (2003), §4.3

```
T_rec_half = lg(p)α + ((p-1)/p)nβ + ((p-1)/p)nγ
```

This is the mirror image of recursive doubling for allgather — halving distances instead
of doubling them. Use for messages **< 512 KB** (commutative operations).

**Pairwise exchange** (long messages):

> "At step i, each process sends data to rank+i, receives data from rank−i, and performs
> the local reduction. The data exchanged is only the data needed for the scattered result
> on the process (≈ n/p)."
>
> — ibid.

```
T_pairwise = (p-1)α + ((p-1)/p)nβ + ((p-1)/p)nγ
```

Same bandwidth requirement as recursive halving, but performs much better for long messages
for the same nearest-neighbor reason as ring allgather. Use for messages **≥ 512 KB**.

**Old MPICH algorithm** for comparison:

```
T_old = (lg p + p - 1)α + (lg p + (p-1)/p)nβ + n·lg(p)·γ
```

Both new algorithms eliminate the `n·lg(p)·β` bandwidth penalty of the old approach.

### 5.5 Rabenseifner's Algorithm for Reduce

Reduce (all-to-one) has a long-message analogue to Van de Geijn broadcast. The key insight:

> "Rabenseifner implements a long-message reduce effectively as a **reduce-scatter
> followed by a gather to the root**, which has the same effect of reducing the bandwidth
> term from n·lg(p)·β to 2nβ."
>
> — Thakur & Gropp (2003), §4.4

```
T_rabenseifner = 2·lg(p)·α + 2·((p-1)/p)·nβ + ((p-1)/p)·nγ
```

Binary tree reduce (old): `T_tree = ⌈lg p⌉(α + nβ + nγ)` — bandwidth scales as n·lg(p)·β.
Rabenseifner: bandwidth scales as 2nβ — a lg(p)/2 improvement, same as Van de Geijn.

**MPICH selection threshold:** binary tree for **≤ 2 KB**, Rabenseifner for **> 2 KB**.

In oneCCL, Rabenseifner is available as `CCL_ALLREDUCE=rabenseifner` (listed as algorithm
ID 6 in Open MPI's `coll_tuned_allreduce_algorithm` parameter). Note: for GPU buffers,
setting `CCL_ALLREDUCE` to any value other than `topo` causes oneCCL to copy data to the
host and run the CPU algorithm. Use `CCL_ALLREDUCE_SCALEOUT=rabenseifner` to select it
for the scaleout phase only while keeping GPU-native scale-up.

### 5.6 One-Shot Allreduce (for GPU Fabrics)

On hardware with direct GPU-to-GPU links (NVLink, AMD Infinity Fabric, Intel UALink/XeLink),
each GPU can simultaneously read from and write to all other GPUs in a single pass:

```
N=4, with direct GPU fabric:
  Each GPU simultaneously pushes its chunk to all 3 peers
  Each GPU locally reduces all received chunks
  → 1 network step, full result everywhere
```

This requires N dedicated peer links per GPU. **Without them**, the single-ported network
assumption means simultaneous fan-out serializes through shared PCIe upstream bandwidth:

> "We assume that the node's network interface is assumed to be single ported; that is,
> at most one message can be sent and one message can be received simultaneously."
>
> — Thakur & Gropp (2003), §3 Cost Model

On NUMA-only hardware, One-Shot's "simultaneous" step becomes serialized. Ring, with its
disciplined one-send/one-receive at any moment, uses the single-ported interface correctly.

---

## 6. Open MPI Algorithm Catalog

Open MPI's `coll_tuned` component implements the full set of algorithms described above.
This table gives you the complete algorithm space that NCCL, RCCL, and oneCCL all draw from:

> **Source:** Open MPI v5.0.x Documentation, *Tuning Collectives*, §11.10.
> [docs.open-mpi.org/en/v5.0.x/tuning-apps/coll-tuned.html](https://docs.open-mpi.org/en/v5.0.x/tuning-apps/coll-tuned.html)

### Allreduce algorithms (Open MPI ID → name)

| ID | Algorithm | When it wins |
|----|---|---|
| 1 | `basic_linear` | Trivial fallback; star topology |
| 2 | `nonoverlapping` | SMP/node-local reduction first |
| 3 | `recursive_doubling` | **Short messages (< 512 KB)**; log₂(p) steps |
| 4 | `ring` | **Long messages (≥ 512 KB)**; nearest-neighbor BW |
| 5 | `segmented_ring` | Very large messages; pipeline across segments |
| 6 | `rabenseifner` | Long messages with reduce; ReduceScatter+Gather |
| 7 | `allgather_reduce` | When allgather bandwidth > reduction cost |

In oneCCL: `CCL_ALLREDUCE=ring` maps to ID 4; `CCL_ALLREDUCE=recursive_doubling` maps to ID 3.

### Allgather algorithms

| Algorithm | When it wins |
|---|---|
| `recursive_doubling` | Short messages; log₂(p) steps |
| `ring` | Long messages; nearest-neighbor BW |
| `bruck` | Non-power-of-two p; ⌈log₂ p⌉ steps |
| `neighbor` | MPI topology-aware (mesh/torus) |

### Broadcast algorithms

| Algorithm | When it wins |
|---|---|
| `binary_tree` | Short messages; log₂(p) latency |
| `binomial` | Non-power-of-two p |
| `pipeline` / `chain` | Very long messages; pipelined segments |
| `scatter_allgather` | Van de Geijn: long messages; 2nβ bandwidth |
| `scatter_allgather_ring` | Van de Geijn with ring phase |
| `knomial` | Tunable fanout (k-nomial tree) |

### Reduce-scatter algorithms

| Algorithm | When it wins |
|---|---|
| `recursive_halving` | Short messages; commutative ops |
| `ring` | Long messages; nearest-neighbor BW |
| `butterfly` | Non-power-of-two p |
| `non_overlapping` | SMP-aware; reduce locally first |

### Alltoall algorithms

| Algorithm | When it wins |
|---|---|
| `linear` | Non-blocking N² pairs; small p, small msg |
| `pairwise` | P blocking rounds; medium messages |
| `modified_bruck` | Small messages; ⌈log p⌉ steps |
| `linear_sync` | Maintains N in-flight pairs; large p |

---

## 7. Message Size Regimes and Algorithm Selection

All real collective libraries use message size to select algorithms. The crossover points
from Thakur & Gropp (2003) are the canonical reference, now directly applicable:

| Collective | Short-message algorithm | Threshold | Long-message algorithm |
|---|---|---|---|
| Allgather | Recursive doubling | **512 KB** | Ring |
| Broadcast | Binary tree | **12 KB** | Van de Geijn (scatter + allgather) |
| Reduce-scatter | Recursive halving | **512 KB** | Pairwise exchange |
| Reduce | Binary tree | **2 KB** | Rabenseifner (RS + gather) |
| Allreduce | Recursive doubling | **~512 KB** | Ring (RS + allgather) |

These thresholds were measured on a Myrinet cluster and IBM SP in 2003. Modern hardware
shifts these numbers — PCIe Gen5 has much higher bandwidth per slot than Myrinet — but
the **algorithm family assignments** (recursive doubling for short, ring for long) remain
correct. oneCCL calibrates its auto thresholds for Intel hardware via its internal tuning.

**Latency-bound vs. bandwidth-bound — the rule of thumb:**

```
T = α·(num_steps) + β·(bytes_per_step_per_link) + γ·(bytes_reduced)

Latency bound:  α term dominates  → minimize number of steps  → Recursive Doubling
Bandwidth bound: β term dominates → minimize bytes on links   → Ring
```

---

## 8. Connecting to oneCCL

With the full model in hand, the oneCCL environment variables map directly:

```python
# For GPU buffers: leave CCL_ALLREDUCE unset (default = "topo", the GPU-native path).
# Control the scaleout algorithm (inter-node) separately:

# Nearest-neighbor ring: T_ring = (p-1)α + ((p-1)/p)nβ
# Bandwidth-optimal. Topology-aware on NUMA when I_MPI_PIN_DOMAIN=socket.
os.environ["CCL_ALLREDUCE_SCALEOUT"] = "ring"

# Recursive doubling: T_rec_dbl = lg(p)α + ((p-1)/p)nβ
# Same bandwidth, fewer steps. Use for small messages only.
os.environ["CCL_ALLREDUCE_SCALEOUT"] = "recursive_doubling"

# Rabenseifner: T = 2·lg(p)·α + 2·((p-1)/p)·nβ + ((p-1)/p)·nγ
# Reduce-scatter + gather. Best bandwidth for reduce-heavy workloads.
os.environ["CCL_ALLREDUCE_SCALEOUT"] = "rabenseifner"

# WARNING: Setting CCL_ALLREDUCE directly (not _SCALEOUT) to any value other than "topo"
# causes oneCCL to copy GPU buffers to host memory and run a CPU algorithm.
# For CPU-only workloads, direct setting is fine:
os.environ["CCL_ALLREDUCE"] = "ring"  # OK for CPU buffers only
```

For TP decode on CRI (NUMA-only, batch=1):
- Message size ≈ 8192 × 2 bytes = **16 KB** → below all crossover thresholds
- Recursive doubling would be the theoretically correct choice (fewer steps for small msg)
- **But** Ring is still recommended for scaleout: the topology-aware nearest-neighbor ring
  avoids cross-socket UPI contention, which recursive doubling's distance-doubling pattern
  cannot exploit. The α-β model assumes all pairs are equal — NUMA violates that assumption.

**Next:** [oneCCL Overview](01_overview) — the oneCCL API surface and where it fits in the stack.

---

## Primary Sources

- Thakur, R. & Gropp, W. (2003). *Improving the Performance of Collective Operations in
  MPICH.* Recent Advances in PVM and MPI, LNCS 2840, pp. 49–60. Full text:
  [wgropp.cs.illinois.edu/bib/papers/pdata/2003/mpicoll-pvmmpi03.pdf](https://wgropp.cs.illinois.edu/bib/papers/pdata/2003/mpicoll-pvmmpi03.pdf)

- Thakur, R., Rabenseifner, R. & Gropp, W. (2005). *Optimization of Collective
  Communication Operations in MPICH.* IJHPCA 19(1), 49–66. The full 2005 paper extends
  this with allreduce, alltoall, and more platforms.

- Open MPI v5.0.x. *Tuning Collectives — coll_tuned component*, §11.10.
  [docs.open-mpi.org/en/v5.0.x/tuning-apps/coll-tuned.html](https://docs.open-mpi.org/en/v5.0.x/tuning-apps/coll-tuned.html)

- MPI Forum. *MPI: A Message-Passing Interface Standard, Version 4.1*, §6, §5.
  [mpi-forum.org](https://www.mpi-forum.org/docs/)
