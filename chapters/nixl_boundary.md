# The oneCCL / NIXL Architectural Boundary

## Why This Boundary Exists

A common mistake in disaggregated inference is routing **all** inter-GPU communication
through oneCCL. This is wrong. oneCCL and NIXL serve different traffic classes:

| Traffic Class | Characteristics | Owner |
|---|---|---|
| TP allreduce | Small, synchronization, latency-critical | **oneCCL** |
| MoE Alltoall | Medium, symmetric, latency-critical | **oneCCL** |
| SP Allgather | Medium, symmetric | **oneCCL** |
| KV cache P→D | Large, directed, point-to-point | **NIXL** |
| KV cache tiering | Large, async, storage-bound | **NIXL** |
| Prompt tokens | Variable, broadcast-like | **NIXL or oneCCL** |

## The Boundary in Code

```python
# oneCCL path (via torch.distributed + ccl backend)
# -- TP layer sync in decode loop
import torch.distributed as dist
dist.all_reduce(activation, op=dist.ReduceOp.SUM)  # oneCCL handles this

# NIXL path (direct NIXL API or via LMCache)
# -- KV cache transfer from prefill to decode node
import nixl
agent = nixl.nixlAgent("decode_worker")
agent.transfer(kv_tensor, dst_rank=decode_rank, ...)  # NIXL handles this
```

## Why NIXL for KV Transport

1. **Size regime**: KV caches are large (GBs at long context). oneCCL Ring Allreduce
   is not designed for this -- it would consume all fabric bandwidth and starve TP collectives.

2. **Directionality**: KV transfer is P2P (prefill rank → specific decode rank),
   not an all-to-all synchronization primitive. NIXL's RDMA/UCX transport is purpose-built for this.

3. **Asynchrony**: KV transfers happen asynchronously with decode compute. NIXL supports
   true async RDMA semantics. oneCCL collectives block until all ranks complete.

4. **Fabric utilization**: Separating the two allows the scheduler to prioritize
   TP allreduce latency (via oneCCL) while batching KV transfers (via NIXL).

## CRI-Specific Note

On CRI nodes (NUMA-only, no XeLink), both oneCCL and NIXL are PCIe-constrained.
This makes the separation even more important -- you cannot afford to have KV bulk
transfers competing with TP allreduce synchronization on the same PCIe uplink.

The recommended approach is to use separate NIC queues / QPs for each traffic class,
which NIXL supports via its transport layer configuration.
