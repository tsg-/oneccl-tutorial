# When to Use Which Collective

This chapter is a decision guide. It maps inference workload patterns to collective choices,
explains the "why" behind each recommendation, and flags the gaps on CRI hardware today.

If you have not read [Foundations](00_foundations) yet, read §3 (what the collectives do) and
§4 (algorithms) first.

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
        See The oneCCL/NIXL Boundary for why.
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
