# The oneCCL / NIXL Architectural Boundary

## Why This Boundary Exists

Disaggregated inference architectures separate **prefill** (prompt processing, compute-heavy)
from **decode** (token generation, memory-bandwidth-heavy) across different node pools. This
creates two fundamentally different communication patterns that require different libraries:

| Traffic Class | Characteristics | Size | Pattern | Owner |
|---|---|---|---|---|
| TP allreduce | Synchronization, latency-critical | 16 KB – 4 MB | All-to-all symmetric | **oneCCL** |
| MoE Alltoall | Expert routing, latency-sensitive | 100 KB – 10 MB | All-to-all asymmetric | **oneCCL** |
| SP Allgather | KV shard assembly | 0.5 – 32 MB | All-to-all symmetric | **oneCCL** |
| KV cache P→D | Bulk transfer, throughput-critical | 100 MB – 10 GB | Point-to-point directed | **NIXL** |
| KV cache tiering | Offload to storage, async | GB-scale | Point-to-point directed | **NIXL** |
| Prompt tokens | Variable, request-routing | 1 KB – 1 MB | Point-to-point or broadcast | **NIXL or oneCCL** |

The fundamental distinction: **oneCCL implements collective synchronization** (all ranks
participate, all ranks block until completion). **NIXL implements directed data movement**
(one sender, one receiver, fully asynchronous, no global synchronization).

Mixing these patterns in a single library creates contention: a large KV transfer on the
same transport as a latency-critical allreduce will delay the allreduce by the transfer
time of the KV cache — potentially milliseconds of added TPOT.

---

## NIXL Architecture

NIXL (NVIDIA Inference Xfer Library) is a lightweight transfer library designed for
point-to-point GPU memory movement. Despite the NVIDIA name, it operates over UCX and
can run on any hardware with a UCX transport provider.

```
┌─────────────────────────────────────────────┐
│              Application Layer              │
│  (vLLM scheduler, TensorRT-LLM, custom)    │
├─────────────────────────────────────────────┤
│              NIXL Agent API                 │
│  nixl_agent → register_memory()            │
│            → get_xfer_descs()              │
│            → initialize_xfer()             │
│            → transfer()                    │
│            → check_xfer_state()            │
├─────────────────────────────────────────────┤
│              NIXL Backend                   │
│  ┌───────────┬───────────┬───────────┐     │
│  │    UCX    │   GDR     │  Storage  │     │
│  │  (RDMA)  │ (GPUDirect)│  (NVMe)  │     │
│  └───────────┴───────────┴───────────┘     │
├─────────────────────────────────────────────┤
│     NIC / Fabric / Storage Hardware         │
└─────────────────────────────────────────────┘
```

Key NIXL concepts:
- **Agent**: A named endpoint that can send or receive data. Each node runs one agent.
- **Memory registration**: GPU or host memory is registered with the transport layer,
  enabling zero-copy RDMA access.
- **Transfer descriptors**: Describe source/destination memory regions for a transfer.
- **Transfer handle**: An opaque handle for an in-flight transfer; polled for completion.
- **Backends**: Pluggable transport backends (UCX for RDMA, GDR for GPUDirect, storage for NVMe).

---

## The Boundary in Code

```python
# oneCCL path (via torch.distributed + ccl backend)
# -- TP layer sync in decode loop
import torch.distributed as dist
dist.all_reduce(activation, op=dist.ReduceOp.SUM)  # oneCCL handles this

# NIXL path (direct NIXL API)
# -- KV cache transfer from prefill to decode node
from nixl import nixl_agent, nixl_agent_config

config = nixl_agent_config(backends=["UCX"])
agent = nixl_agent("decode_worker", config)

# Register memory, build transfer descriptors, then initiate transfer
reg = agent.register_memory(kv_tensor)
local_descs = agent.get_xfer_descs([kv_tensor])
xfer_handle = agent.initialize_xfer("READ", local_descs, remote_descs, "prefill_node", "done")
state = agent.transfer(xfer_handle)  # NIXL handles this — async RDMA
```

---

## Why NIXL for KV Transport

### 1. Size Regime Mismatch

KV caches for long-context models are massive:

```
Llama-3 70B, context=4096, TP=4:
  Layers: 80
  KV heads per TP shard: 8 (GQA)
  Head dim: 128
  KV per layer: 2 × 8 × 4096 × 128 × 2 bytes (BF16) = 16 MB
  Total KV cache: 80 × 16 MB = 1.28 GB per TP shard

Llama-3 70B, context=32768:
  Total KV cache: 80 × 128 MB = 10.24 GB per TP shard
```

Routing 1-10 GB through oneCCL Ring Allreduce would:
- Consume all PCIe bandwidth for 40-400 ms (at 25 GB/s)
- Block all 160 TP allreduces during that time (stalling decode for the entire cluster)
- Require all ranks to participate (even ranks that don't need the KV data)

NIXL transfers are point-to-point and non-blocking — they consume bandwidth only between
the sender and receiver, and other ranks continue computing.

### 2. Directionality

KV transfer is inherently P2P: one prefill rank computed the KV cache, one decode rank
needs it. This is not a collective pattern — forcing it into `allgather` or `broadcast`
wastes (N-1)/N of the bandwidth on data that N-1 ranks immediately discard.

### 3. Asynchrony

NIXL transfers are fire-and-forget with completion polling:

```python
# Non-blocking: returns immediately
state = agent.transfer(xfer_handle)

# Decode loop continues while KV transfers in background
while state != "done":
    # Process other requests, run decode steps
    state = agent.check_xfer_state(xfer_handle)
```

oneCCL collectives are fundamentally **synchronous barriers** — even with `async_op=True`,
all ranks must eventually call the collective and the result is only valid after all ranks
complete. This barrier semantics is correct for TP (you need all partial sums before
continuing) but wrong for KV transfer (only one rank needs the data).

### 4. Fabric Utilization and QoS

Separating traffic classes enables priority scheduling at the NIC level:

```
NIC Queue 0 (high priority, low latency):
  TP allreduce messages (16 KB, sub-50µs SLO)
  MoE alltoall (100 KB–10 MB)

NIC Queue 1 (bulk, throughput-optimized):
  KV cache transfers (GB-scale, 10-100ms acceptable)
  Model weight distribution (startup only)
```

NIXL supports explicit queue pair (QP) selection via its backend configuration.
oneCCL uses its own QPs for collective traffic. By keeping them separate, a large
in-flight KV transfer cannot head-of-line block a latency-critical allreduce.

---

## Worked Example: Transfer Time Budget

For a disaggregated Llama-3 70B deployment with continuous batching:

```
KV cache per request (ctx=4096, TP=4): 1.28 GB
Network: 200 Gbps (25 GB/s) per NIC, 2 NICs per node

Transfer time (1 NIC): 1.28 GB / 25 GB/s = 51 ms
Transfer time (2 NICs, striped): 1.28 GB / 50 GB/s = 26 ms

Decode TPOT target: 50 ms/token
TP comm budget per token: ~5 ms (10%)

If KV transfer shared the same NIC queues as TP allreduce:
  During the 26-51ms KV transfer window, TP allreduce latency
  would spike from ~3ms to 20-50ms (head-of-line blocking)
  → 10-100 tokens generated with degraded TPOT
```

This is why physical separation (different QPs, or different NICs) is essential, not
optional.

---

## CRI-Specific Considerations

On CRI nodes (NUMA-only, no XeLink), both oneCCL and NIXL are PCIe-constrained.
This makes the separation even more important — you cannot afford to have KV bulk
transfers competing with TP allreduce synchronization on the same PCIe uplink.

The PCIe Gen5 x16 bandwidth ceiling is ~32 GB/s per direction per root port. With
4 GPUs sharing a root complex:

```
Available PCIe bandwidth per root complex: ~32 GB/s
TP allreduce demand (4 GPUs, ring, 16 KB msg): negligible BW, latency-bound
KV transfer demand (1.28 GB at max rate): saturates link for 40ms

If both share a root complex: KV transfer starves allreduce of PCIe slots
Solution: route KV transfers through a separate NIC on a different PCIe root
```

The recommended deployment:
- **NIC 0** (Socket 0 PCIe root): TP/EP collective traffic via oneCCL
- **NIC 1** (Socket 1 PCIe root): KV cache bulk transfers via NIXL

This ensures the two traffic classes never contend for the same physical PCIe link.
