# When to Use Which Collective

This chapter is a decision guide for both inference and training. It maps workload patterns
to collective choices, explains the "why" behind each recommendation, and flags the gaps on
CRI hardware today.

If you have not read [Foundations](00_foundations) yet, read §3 (what the collectives do) and
§4 (algorithms) first.

**Navigation:** Jump directly to [Training Workloads](#training-workloads) if you are here
for training (DDP, ZeRO, pipeline parallelism). The inference sections are first; the
training decision tree, ZeRO comparison, and training anti-patterns are at the bottom.

---

## Decision Framework

Before calling any collective, answer these five questions. The answers determine everything.

```
1. Do all ranks need the result, or just one?
   └── All ranks need it → Allreduce / Allgather / Alltoall
   └── One rank needs it → Reduce / Gather / Scatter

2. Are you combining values (summing) or just distributing data?
   └── Combining: Allreduce (keep same size) or Reduce-Scatter (split result)
   └── Distributing: Allgather (each contributes distinct chunk)

3. Is the traffic symmetric (same amount per rank) or asymmetric?
   └── Symmetric: allreduce, allgather, alltoall
   └── Asymmetric: alltoallv, allgatherv (variable-length variants)

4. What is the message size?
   └── < 512 KB → latency-critical; recursive doubling wins (fewer steps)
   └── ≥ 512 KB → bandwidth-critical; ring wins (nearest-neighbor BW)
   (These are the measured MPICH crossover thresholds: Thakur & Gropp 2003/2005.
    See Foundations §6 for the exact per-collective breakdown.)
   Note for training: all gradient collectives are far above this threshold
   (GB-scale messages). Ring is always correct for training; skip this question.

5. Does the result need to be on every rank, or can ranks hold different shards?
   └── Everyone needs full result → allreduce (not reduce-scatter)
   └── Sharding is fine → reduce-scatter (cheaper by 2x on large tensors)
```

---

## Collective Decision Tree for Inference

```
What are you synchronizing?
│
├── TP partial activations after linear layer
│   └── → Allreduce (SUM)
│       All ranks computed a partial output. Sum them. All ranks need the full result.
│       Size: (batch × seq × hidden) × 2 bytes (BF16)
│       For batch=1, seq=1, hidden=8192: 16 KB  ← small, latency-critical
│       For batch=32, seq=512, hidden=8192: 256 MB ← use ReduceScatter+Allgather instead
│
├── Sequence sharding before attention (SP)
│   └── → Allgather
│       Each rank holds seq/N tokens. Need full context for attention.
│       Size grows with sequence length — may be large. Ring (pipelined RS+AG) on CRI.
│
├── MoE expert routing — tokens to experts
│   └── → Alltoall (or Alltoallv)
│       Each rank sends different tokens to different expert-owning ranks.
│       Alltoall if token counts are uniform (balanced routing).
│       Alltoallv if counts vary (imbalanced routing — use this one in practice).
│       Size: (tokens per rank × expert) × hidden_dim × 2 bytes
│
├── Recovering sharded MoE output
│   └── → Alltoall (reverse of routing)
│       After expert compute, outputs need to be routed back to their origin ranks.
│       Same pattern as routing, transposed.
│
├── Gradient aggregation (training, not inference)
│   └── → Allreduce (SUM) or ReduceScatter + Allgather
│       If using ZeRO-style optimizer: ReduceScatter to reduce memory footprint.
│       If not: Allreduce is simpler.
│
├── Sync a scalar/flag across all ranks (e.g., EOS token detection)
│   └── → Allreduce (MAX or OR-as-SUM)
│       Tiny message (<< 1 KB). Latency dominates. Any algorithm works.
│       Consider: dist.all_reduce(flag_tensor, op=dist.ReduceOp.MAX)
│
├── Broadcast model weights or config at startup
│   └── → Broadcast
│       One rank has the source data; all others need it.
│       Only use at initialization — not in the decode hot path.
│
└── KV cache transfer (Prefill → Decode nodes)
    └── → NOT oneCCL. Use NIXL.
        See [The oneCCL/NIXL Boundary](nixl_boundary) for why.
```

---

## Allreduce vs. Reduce-Scatter + Allgather

For large tensors in TP, you have a choice:

| Approach | What Happens | Memory After | Best For |
|---|---|---|---|
| `allreduce` | Full tensor reduced, full tensor on every rank | Full tensor × N_ranks total | Small TP shards (<256KB per rank) |
| `reduce_scatter` | Reduce then scatter: rank i gets chunk i of result | Full tensor across cluster (1/N per rank) | Large tensors, ZeRO-style sharding |
| `reduce_scatter` + `allgather` | Equivalent to allreduce but pipelined | Full tensor everywhere | Large tensors, memory-bandwidth tradeoff |

For decode (batch=1, seq=1) the tensors are tiny. `allreduce` is correct and simpler.
For large prefill batches, consider `reduce_scatter` followed by `allgather` to pipeline
the reduction with local compute.

---

## The "v" Variants: When to Use alltoallv and allgatherv

The base variants (`alltoall`, `allgather`) require all ranks to send/receive the same count.
The "v" variants add per-rank count and displacement arrays.

```python
# alltoall: rank i sends `count` elements to every other rank
dist.all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None)
# → equal split implied

# alltoallv: rank i sends different amounts to each peer
output_split_sizes = [n_tokens_for_rank_j for j in range(world_size)]  # what I receive
input_split_sizes  = [n_tokens_from_rank_j for j in range(world_size)] # what I send
dist.all_to_all_single(output, input, output_split_sizes, input_split_sizes)
```

**MoE inference always uses alltoallv.** Token routing is never perfectly balanced. If you
use the fixed-size `alltoall` with imbalanced routing, you waste bandwidth on padding.

### Worked Example: MoE Buffer Sizing

Consider Mixtral 8×7B with 8 experts distributed across 4 GPUs (2 experts per GPU):

```
Model parameters:
  Hidden dim: 4096
  Expert FFN dim: 14336
  Top-K routing: 2 (each token activates 2 of 8 experts)
  Batch: 32 tokens in decode step

Token routing (example — worst case skew):
  Expert 0 (GPU 0): receives 12 tokens
  Expert 1 (GPU 0): receives 4 tokens
  Expert 2 (GPU 1): receives 8 tokens
  Expert 3 (GPU 1): receives 6 tokens
  Expert 4 (GPU 2): receives 10 tokens
  Expert 5 (GPU 2): receives 8 tokens
  Expert 6 (GPU 3): receives 10 tokens
  Expert 7 (GPU 3): receives 6 tokens
  Total activations: 32 × 2 = 64 (each token goes to 2 experts)

Dispatch alltoallv buffer sizes:
  Each token sends: hidden_dim × sizeof(BF16) = 4096 × 2 = 8 KB
  GPU 0 sends to GPU 1: (tokens routed to experts 2,3 from GPU 0's batch) × 8 KB
  GPU 0 sends to GPU 2: (tokens routed to experts 4,5 from GPU 0's batch) × 8 KB
  ... (asymmetric per rank)

  Peak send buffer per GPU: ~32 tokens × 8 KB = 256 KB
  Peak recv buffer per GPU: ~16 tokens × 8 KB = 128 KB (2 experts, max load)

Combine alltoallv (reverse direction):
  Each expert output: hidden_dim × sizeof(BF16) = 8 KB per token
  Same asymmetric pattern, transposed

Total alltoallv data per MoE layer: ~512 KB – 1 MB (both directions)
With 8 MoE layers in a 32-layer model: ~4-8 MB total MoE comm per forward pass
```

The key insight: MoE alltoallv messages are **medium-sized** (100 KB–1 MB per rank pair)
and **asymmetric**. The `topo` algorithm handles this correctly — it uses scatter for the
scaleout phase, which supports variable-length messages natively.

> **Reference:** MPICH documentation, *MPI_Alltoallv*, covers the variable displacement
> semantics. The pattern of (send_counts, send_displs, recv_counts, recv_displs) maps
> directly to PyTorch's `input_split_sizes` / `output_split_sizes`.

---

## Anti-Patterns: When Collectives Are Wrong

These situations are commonly mishandled:

| Situation | Wrong Choice | Right Choice |
|---|---|---|
| KV cache P→D transfer | allreduce or alltoall | NIXL (RDMA, async, P2P) |
| Large directed tensor send (one rank to one rank) | allgather | Point-to-point (dist.send/recv) or NIXL |
| Prefill node feeding decode node | broadcast | NIXL — broadcast would block all ranks |
| Data loading / checkpoint save | any collective | Avoid entirely; use filesystem |
| Per-layer debugging output | allgather of activations | Do it on single rank only; conditionally |

The core principle: **oneCCL is for synchronization primitives across TP/EP ranks.
NIXL is for directed data movement between prefill and decode nodes.**

---

## CRI Hardware Gaps and Workarounds

The following collectives have known gaps on CRI (NUMA-only, as of mid-2025):

| Collective | Gap | Workaround |
|---|---|---|
| Scatter (intra-node) | Ring implementation in P0 | For small data: use Broadcast + local select. For KV: use NIXL. |
| Gather (intra-node) | Same as Scatter, P0 | Use Reduce-Scatter (and discard unused chunks) |
| Alltoall (inter-node, scale-out) | Direct/Ring algorithms missing; topo+scatter only | Use topo (it works); file a bug if latency is unacceptable |
| Reduce-Scatter (async) | Async variants less tested on XPU | Test with sync first; file bug with repro if async differs |

For anything not in the "Done" column in [the overview table](../intro), verify with a
correctness test before deploying.

---

## Quick Reference: Inference Collective Cheat Sheet

| Inference Pattern | Collective | Size Regime | Algorithm (CRI) |
|---|---|---|---|
| TP decode (batch=1) | Allreduce | < 64 KB | Ring (small-msg) |
| TP decode (batch=16+) | Allreduce | 64 KB–4 MB | Ring (pipelined RS+AG) |
| TP prefill (long ctx) | Allreduce or ReduceScatter | > 1 MB | Ring |
| SP attention gather | Allgather | 0.5–32 MB | Ring (pipelined) |
| MoE routing (uniform) | Alltoall | varies | topo (scale-up) + scatter (scaleout) |
| MoE routing (skewed) | Alltoallv | varies | topo (scale-up) + scatter (scaleout) |
| Control sync (EOS) | Allreduce MAX | < 1 KB | Any |
| KV cache P→D | — | GB-scale | **NIXL** (not oneCCL) |

---

## Training Workloads

Training uses the same collectives as inference but in different configurations. The critical
difference is **message size regime**: inference decode is latency-bound (16 KB, 160×/token),
while training gradient sync is bandwidth-bound (GB-scale, 1×/step). The algorithm families
are the same — ring for large messages, recursive doubling for small — but the right operating
point shifts entirely to the ring side.

### Training Collective Decision Tree

```
What are you synchronizing?
│
├── Data-parallel gradient aggregation (DP, DDP, ZeRO-0/1)
│   └── → Allreduce (SUM)
│       Every GPU computed gradients on its batch shard. Sum to get full-dataset gradient.
│       Size: (model params) × bytes/param
│       7B model (BF16): 14 GB total — partitioned by layer, streamed with backward pass
│       Algorithm: Ring (always bandwidth-bound)
│       ZeRO-1 note: gradient communication is unchanged (still Allreduce); ZeRO-1 only
│             shards optimizer state, which is local CPU work after the Allreduce completes
│
├── ZeRO-2 gradient partitioning (optimizer sharding)
│   └── → ReduceScatter (SUM)
│       Like allreduce but each rank keeps only 1/N of the gradient (its own shard).
│       Each rank only updates its own optimizer state shard — no need for full gradient on all ranks.
│       Size per rank: model_params/N × 2 bytes
│       Algorithm: Ring reduce-scatter (pairwise exchange for large messages)
│
├── ZeRO-3 parameter reconstruction (parameter sharding)
│   ├── Before forward pass: → Allgather
│   │   Each rank holds 1/N of parameters. Reconstruct full layer before the matmul.
│   │   Discard the gathered copy immediately after the layer to free memory.
│   │   Size per allgather: one layer's parameters (varies: 100 MB for 70B attention)
│   │   Algorithm: Ring allgather
│   └── After backward pass: → ReduceScatter
│       Reduce gradients and scatter — same as ZeRO-2.
│
├── Pipeline parallelism activation handoff (PP)
│   └── → NOT a collective. Use point-to-point send/recv.
│       Stage i sends its output activation to stage i+1. Stage i+1 sends gradients back.
│       This is P2P: one sender, one receiver. Forcing it into allgather wastes (N-1)/N BW.
│       In PyTorch: dist.send(tensor, dst=next_rank) / dist.recv(tensor, src=prev_rank)
│       Async variants: dist.isend() / dist.irecv() with req.wait() for overlap
│
├── Tensor parallel gradient sync (TP, same as inference)
│   └── → Allreduce (SUM) — same as inference TP layer sync
│       During training the backward pass needs the same allreduce as the forward pass.
│       Message sizes identical to forward pass for the same layer.
│
├── Expert parallel gradient sync (EP, MoE training)
│   └── → Alltoallv (forward + backward)
│       Forward: route tokens to expert GPUs (same as inference MoE dispatch)
│       Backward: route gradients back — same alltoallv with transposed send/recv counts
│       Plus a separate ReduceScatter on each expert's gradients across DP dimension
│
├── Model weight broadcast at startup / checkpoint restore
│   └── → Broadcast
│       Rank 0 loads checkpoint, broadcasts to all other ranks.
│       Only at startup — never in training hot path.
│       Size: full model (GB-scale). Algorithm: Van de Geijn (scatter+allgather ring)
│
└── Gradient norm computation (for gradient clipping)
    └── → Allreduce (SUM of squared norms, then local sqrt)
        Each rank computes local_norm² = sum(grad²). Allreduce sums across ranks.
        Then each rank does sqrt(total) locally and clips.
        Size: scalar (8 bytes). Any algorithm. Latency dominates.
```

### ZeRO Stage Comparison

Understanding ZeRO is understanding which collectives run when:

```
ZeRO Stage 0 (no sharding, baseline DDP):
  Forward:   no collective
  Backward:  Allreduce (gradients, full model, every step)
  Per-rank memory: full model + full gradients + full optimizer state

ZeRO Stage 1 (optimizer state sharding):
  Forward:   no collective
  Backward:  Allreduce (gradients, full model) — same as stage 0
  Optimizer: each rank updates its shard only
  Per-rank memory: full model + full gradients + 1/N optimizer state
  Collective pattern change: none in the critical backward path

ZeRO Stage 2 (gradient + optimizer state sharding):
  Forward:   no collective
  Backward:  ReduceScatter (replaces Allreduce — each rank reduces its shard only)
  Optimizer: each rank updates its shard only
  Per-rank memory: full model + 1/N gradients + 1/N optimizer state
  Collective change: backward becomes ReduceScatter (not Allreduce)
  Savings: gradient buffer drops from M to M/N

ZeRO Stage 3 (parameter + gradient + optimizer sharding):
  Forward:   Allgather (per layer, reconstruct + discard after each layer)
  Backward:  ReduceScatter (per layer, reduce + discard after each layer)
  Optimizer: each rank updates its shard only
  Per-rank memory: 1/N model + 1/N gradients + 1/N optimizer state
  Collective change: adds Allgather on critical forward path
  Cost: 2× allreduce bandwidth equivalent (ReduceScatter + Allgather), but now memory-
        optimal. For large models that don't fit otherwise, this is the right trade.
```

### Training Message Sizes vs Inference: The Algorithm Crossover

The same message size thresholds from §7 of Foundations apply, but training lands
on the opposite side of all of them:

| Workload | Example Message | Regime | Dominant Algorithm |
|---|---|---|---|
| TP decode (inference) | hidden=8192, BF16, batch=1 | **16 KB** — latency-bound | Recursive doubling (scaleout fallback) |
| TP prefill (inference) | hidden=8192, BF16, batch=32, seq=128 | **64 MB** — bandwidth-bound | Ring |
| ZeRO-2 ReduceScatter (training) | 7B model / N_ranks | **GB-scale** — deep BW-bound | Ring |
| ZeRO-3 Allgather (training) | one layer / N_ranks | **100 MB–1 GB** — BW-bound | Ring |
| DP Allreduce (training, DDP) | 7B model BF16 | **14 GB total** (streamed per layer) | Ring |

**For training on CRI:** ring is always correct. The 512 KB threshold from Foundations §7
is irrelevant — training collective messages are 3-4 orders of magnitude larger. The
important configuration is ensuring `CCL_ALLREDUCE_SCALEOUT=ring` and that NUMA pinning
is correct so intra-node ring steps stay on-socket.

### Pipeline Parallelism: Not a Collective

Pipeline parallelism is a common source of confusion. **It does not use collectives.**

```
PP=4, forward pass, microbatch flow:

  Stage 0 (Rank 0)  →  Stage 1 (Rank 1)  →  Stage 2 (Rank 2)  →  Stage 3 (Rank 3)
  Layers 0-19           Layers 20-39          Layers 40-59          Layers 60-79

  Send: dist.isend(activation, dst=1)    ← sends to next stage only
  Recv: dist.irecv(activation, src=0)    ← receives from prev stage only

  Backward:
  Stage 3 → Stage 2 → Stage 1 → Stage 0  (gradients flow backward)
```

`dist.isend`/`dist.irecv` are point-to-point operations — they involve exactly two ranks.
A collective like Allgather would broadcast to all N stages, which is wrong and wastes
(N-1)/N of the bandwidth. oneCCL supports P2P via `dist.send`/`dist.recv` backed by the
CCL transport layer; these are fine to use alongside collectives in the same process group.

### Training Anti-Patterns

| Situation | Wrong | Right |
|---|---|---|
| Gradient sync across DP ranks | Allgather (don't need full gradient on all ranks) | Allreduce or ReduceScatter (ZeRO-2+) |
| PP stage activation handoff | Broadcast or Allgather | `dist.isend`/`dist.irecv` (P2P) |
| ZeRO-3 forward pass | Broadcast from rank 0 | Allgather (every rank holds 1/N and reconstructs together) |
| Layer-wise gradient accumulation (gradient checkpointing) | One allreduce per micro-step | Accumulate locally, one allreduce per global step |
| Loading a checkpoint during training | Broadcast (blocks all ranks for GB-scale load) | Load on one rank from filesystem, then Broadcast; or use NIXL for large shards |

### Training and NIXL

Training **does not use NIXL** in the standard case. NIXL is specific to disaggregated
inference architectures (prefill→decode KV transfer). During training, all communication
is collective and stays within oneCCL. The oneCCL/NIXL boundary described in
[The oneCCL/NIXL Boundary](nixl_boundary) does not exist in training deployments — the
NIXL column is simply absent.

The only training-adjacent case where NIXL could appear is model checkpoint streaming
during long runs (loading/saving weight shards asynchronously while training continues),
but that is application-level engineering, not collective communication.

---

## Quick Reference: Training Collective Cheat Sheet

| Training Pattern | Collective | Size Regime | Algorithm (CRI) |
|---|---|---|---|
| DDP gradient sync | Allreduce | GB-scale (per-layer stream) | Ring |
| ZeRO-2 backward | ReduceScatter | GB-scale / N_ranks | Ring (pairwise exchange) |
| ZeRO-3 forward | Allgather (per layer) | ~100 MB–1 GB / N_ranks | Ring |
| ZeRO-3 backward | ReduceScatter (per layer) | ~100 MB–1 GB / N_ranks | Ring |
| MoE training dispatch | Alltoallv | varies (asymmetric) | topo (scale-up) + scatter (scaleout) |
| Pipeline parallelism | P2P send/recv | activation size | dist.isend/irecv |
| Grad norm (clipping) | Allreduce (SUM) | 8 bytes | Any |
| Weight broadcast (startup) | Broadcast | Full model (GB) | Van de Geijn (scatter+allgather) |

**Ready to run code?** Start with [Environment Setup](../notebooks/02_environment_setup),
then work through the notebooks in order:
[Allreduce](../notebooks/03a_allreduce_walkthrough) →
[Allgather](../notebooks/03b_allgather) →
[Alltoall / MoE](../notebooks/03c_alltoall_moe) →
[End-to-End TP Decode](../notebooks/04_inference_tp_decode)
