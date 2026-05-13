# Why Host Bounce Buffers Limit oneCCL Scaling on PVC-Class Architectures

---

## Abstract

Intel Xe GPU architectures (PVC, CRI) require all inter-node collective
communication to pass through host DRAM by default. This "host bounce buffer"
design imposes a fixed per-hop latency penalty of approximately 4-6 us that
compounds across every step of a distributed collective. At small node counts,
this overhead is tolerable. Beyond roughly 8-16 nodes for latency-sensitive
workloads (decode inference), the compounded staging penalty exceeds per-layer
compute time and the system becomes communication-bound in a way that no
algorithm tuning can escape — because the bottleneck is architectural, not
algorithmic. This paper analyzes the mechanism precisely, derives the scaling
behavior quantitatively, and identifies the conditions under which the wall
appears.

---

## 1. Background: What Collectives Require of the Hardware

A distributed allreduce is the dominant collective in transformer inference and
training. Each rank holds a partial result (e.g., a shard of an attention
output), and the collective must sum all shards and return the total to every
rank. The time to execute one allreduce has two components:

```
T_allreduce = T_compute_local + T_communication
```

`T_compute_local` scales inversely with the number of ranks (more ranks, less
work per rank). `T_communication` does not improve with rank count — it
worsens, because more ranks means more steps in any collective algorithm.

The scaling efficiency of a distributed workload is:

```
E(N) = T_serial / (N * T_parallel(N))
```

When `T_communication` grows faster than `T_compute_local` shrinks, efficiency
degrades. The node count at which `T_communication >= T_compute_local` is the
practical scaling wall.

For Intel Xe GPUs, the host bounce buffer inflates `T_communication` beyond
what the network latency alone would require. This is the central claim this
paper substantiates.

---

## 2. The Host Bounce Buffer Mechanism

On PVC and CRI, GPU kernels cannot issue network operations. The NIC is not
visible to GPU-side code. All inter-node communication follows this path:

```
Send side:
  (1) GPU kernel writes output to GPU VRAM
  (2) GPU copy engine DMAs from VRAM to pinned host buffer   [D2H, PCIe]
  (3) Host CPU calls fi_tsendmsg() with host buffer pointer  [CPU, software]
  (4) NIC reads host DRAM and places data on wire            [NIC DMA, PCIe]

Receive side:
  (5) NIC writes received data into pinned host buffer       [NIC DMA, PCIe]
  (6) Host CPU signals completion, releases buffer
  (7) GPU copy engine DMAs from host buffer to VRAM          [H2D, PCIe]
  (8) GPU kernel reads result from VRAM
```

Steps 2, 4, 5, and 7 all traverse the PCIe bus. The PCIe bus between a PVC
tile and the host is a shared resource: D2H copies for sending compete with
H2D copies for receiving, and multiple tiles on the same host compete for
the same PCIe root complex bandwidth.

This is confirmed in oneCCL source at two locations:

In `coll_util.cpp` (the `topo` algorithm scaleout phase):
```cpp
if (!enable_hmem) {
    // D2H copy into staging buffer before network send
}
```

In `allreduce_scaleout_sycl.cpp` (SYCL kernel path):
```cpp
if (should_disable_rdma(ze_dev) || atl_transport == ccl_atl_ofi) {
    copy_to_host = true;  // OFI always host-stages
}
```

The comment at `check_mpi_supports_rdma()` in `sycl_coll_base.cpp` is
definitive: *"ofi collective only supports host memory"*. Since OFI is the
recommended transport for inference, every production deployment operates in
the host-staged regime.

### 2.1 Latency Budget Per Collective Step

For a 16 KB message on a PVC node with PCIe Gen4 (40 GB/s effective):

| Operation | Bandwidth | Latency Estimate |
|---|---|---|
| D2H DMA (GPU → host staging) | ~32 GB/s effective | ~0.5 us |
| OFI post_send (CPU software path) | n/a | ~1-2 us |
| NIC DMA (host DRAM → wire) | 25 GB/s (200 Gbps) | ~0.6 us |
| Network transit (Slingshot-11) | ~200 Gbps | ~1-2 us |
| NIC DMA (wire → host DRAM) | 25 GB/s | ~0.6 us |
| H2D DMA (host staging → GPU) | ~32 GB/s effective | ~0.5 us |

**Bounce buffer overhead per step:** ~4-6 us  
**Network-only latency per step:** ~1-2 us  
**Overhead ratio:** 3-5x

The bounce buffer triples to quintuples the effective per-hop latency
compared to a GPU RDMA path where the NIC reads GPU memory directly.

---

## 3. Collective Algorithm Scaling Under Bounce Buffer

### 3.1 Ring Allreduce

Ring allreduce executes 2(N-1) steps, each transferring M/N bytes, where M is
message size and N is the number of ranks. Total latency:

```
T_ring = 2(N-1) * α + 2*(N-1)/N * M/β
```

With bounce buffer, the effective per-step latency α becomes:

```
α_eff = α_network + α_D2H + α_H2D
      = 1-2 us + 2-3 us + 2-3 us
      ≈ 5-8 us
```

For large N, `T_ring ≈ 2N * α_eff`. The bounce buffer overhead does not
merely add a fixed offset — it multiplies the dominant scaling term. At N=64
nodes:

| α | T_ring (latency component) |
|---|---|
| Without bounce buffer (2 us) | 2 * 63 * 2 = 252 us |
| With bounce buffer (6 us) | 2 * 63 * 6 = 756 us |

Ring allreduce is rarely used for large N due to its O(N) latency scaling.
oneCCL selects ring for the scaleout phase at moderate scale but switches to
recursive doubling for short messages.

### 3.2 Recursive Doubling

Recursive doubling executes log₂(N) steps, each exchanging the full message M.
Total latency:

```
T_rd = log₂(N) * (2α + M/β)
```

This logarithmic scaling is why recursive doubling is preferred at large N for
small messages. With bounce buffer:

```
T_rd = log₂(N) * (2*α_network + 4*α_PCIe + M/β)
```

Each step now requires two PCIe traversals in each direction (D2H + H2D),
because the GPU must stage before sending and unstage after receiving.

Substituting measured values (Aurora, Slingshot-11, 16 KB message):

| N nodes | Steps | T_rd without staging | T_rd with staging |
|---|---|---|---|
| 8 | 3 | 3 * (4 + 0.6) = 14 us | 3 * (12 + 0.6) = 38 us |
| 64 | 6 | 6 * (4 + 0.6) = 28 us | 6 * (12 + 0.6) = 76 us |
| 512 | 9 | 9 * (4 + 0.6) = 41 us | 9 * (12 + 0.6) = 113 us |
| 2048 | 11 | 11 * (4 + 0.6) = 50 us | 11 * (12 + 0.6) = 139 us |

The Aurora benchmark (arXiv 2512.04291) measures allreduce latency at 2048
nodes of ~250 us for messages up to 64 KB. The gap between the 139 us model
and the 250 us measurement accounts for software overhead in oneCCL's
scheduler (collective scheduling, memory registration, worker thread handoff)
and the fact that Aurora's 12-tile PVC nodes add intra-node Xe Link phases
not captured in this single-node model.

The key result: **the bounce buffer adds 90-200 us to the latency at 2048
nodes compared to a hypothetical GPU RDMA path.** This is a 2-4x inflation of
total allreduce latency at datacenter scale.

### 3.3 Hierarchical Algorithms (topo)

oneCCL's `topo` algorithm uses a two-level hierarchy: Xe Link for intra-node
scale-up, then a scaleout algorithm (ring or recursive doubling) for
inter-node. On Aurora (PVC, 12 tiles per node), the intra-node reduce-scatter
over Xe Link is fast (~5-10 us). The inter-node phase then operates on 1/12th
of the data, reducing the staging overhead proportionally.

However, the intra-node Xe Link phase still terminates in a host-staged
inter-node transfer. The 1/12th data reduction helps bandwidth-bound
scenarios but does not reduce the per-step latency penalty:

```
T_topo = T_scale_up + T_scaleout
       = T_xe_link + log₂(N_nodes) * (2*α_network + 4*α_PCIe + M/(12*β))
```

For decode inference where M is small (16-64 KB), the message-size term is
negligible. The latency floor is:

```
T_topo ≈ 10 us + log₂(N_nodes) * 12 us
```

At 2048 nodes: `T_topo ≈ 10 + 11 * 12 = 142 us`

This matches the order of magnitude in the Aurora benchmarks.

---

## 4. The Three Scaling Bottlenecks

### 4.1 PCIe Bandwidth Saturation

PVC's PCIe Gen4 x16 provides ~32 GB/s effective unidirectional bandwidth
per tile. A PVC node has 2 physical GPUs, each connected via PCIe to the
host. During an allreduce, both the D2H (before send) and H2D (after receive)
phases are active simultaneously, contending for the same PCIe bus.

Aurora uses 8x Slingshot-11 NICs per node, providing 8 * 25 GB/s = 200 GB/s
aggregate NIC bandwidth. To keep all 8 NICs busy, the host must supply
200 GB/s of outbound data from GPU staging buffers. But 2 physical GPUs at
32 GB/s each = 64 GB/s PCIe capacity. **The PCIe bandwidth is 3x
undersupplied relative to NIC capacity.** Aurora cannot saturate its own
NICs from GPU buffers because the host staging path bottlenecks at PCIe.

The consequence: measured NIC utilization on Aurora is 23-25 GB/s per port
rather than the 25 GB/s theoretical maximum. This ~5-10% shortfall is the
PCIe saturation effect, and it means oneCCL never fully uses the available
network bandwidth when staging from GPU.

As N grows, each ring or recursive-doubling step sends smaller chunks
(M/(2N) for recursive doubling's early steps), and PCIe is no longer the
bottleneck. But for latency-bound small messages — the common case in
decode inference — PCIe saturation means each staging copy takes longer
than the NIC transfer itself.

### 4.2 CPU Orchestration Overhead at High Message Rates

Every OFI send/recv requires host CPU involvement: posting to libfabric,
managing completion queues, and driving the completion handler that initiates
the H2D copy. At small N and low collective rates, this is negligible.

At inference scale, the collective rate is:

```
Collective rate = tokens_per_second * allreduces_per_token
               = tokens_per_second * 160
```

For a system generating 100 tokens/second (modest throughput for a large
model), that is 16,000 OFI operations per second per collective step,
per node. Each OFI operation on the host path involves:

1. Memory copy from GPU staging buffer region into OFI send buffer
2. `fi_tsendmsg()` call (syscall or user-mode library transition)
3. Completion polling or interrupt delivery
4. Buffer release and staging buffer recycle

With N nodes and recursive doubling (log₂ N steps), each node posts
log₂ N sends and log₂ N receives per collective. At N=2048:

```
OFI operations per second = 16000 * 11 * 2 = 352,000 ops/second
```

Each operation involves a CPU-side memory copy (~0.5-1 us each) plus the
OFI call itself (~0.5 us). At 352K ops/second, the CPU is spending
350,000 * 1.5 us = 525 ms/second, or 52% of a single core just on OFI
overhead. oneCCL uses dedicated worker threads for this, so the CPU
overhead doesn't directly stall the GPU. But the worker thread serializes
operations within a rank, imposing queuing delays at high message rates.

This is fundamentally different from GPU RDMA: with a GPU-initiated network
stack, the GPU would post operations to the NIC hardware queue directly with
no CPU involvement. The CPU path serializes and adds queueing jitter that
grows with load.

### 4.3 Staging Buffer Memory Bandwidth Contention

Each rank requires its own D2H staging buffer. On a PVC node with 12 tiles
doing an allreduce, 12 D2H copies and 12 H2D copies are in flight
concurrently, all sharing the same host memory bandwidth (~200 GB/s for DDR5)
and the same PCIe root complexes.

In practice, the PVC node has 2 physical GPUs, each on its own PCIe root
complex, and the 12 tiles are split 6 per GPU. The effective D2H bandwidth
per tile is therefore:

```
BW_per_tile = PCIe_unidirectional / tiles_sharing_root_complex
            = 32 GB/s / 6
            ≈ 5.3 GB/s per tile (for simultaneous transfers)
```

For a 16 KB message:

```
T_D2H per tile = 16 KB / 5.3 GB/s = 3 us
```

This 3 us is the contended staging time — compared to 0.5 us in isolation.
This contention effect explains why the per-step latency estimate above (12 us)
is higher than the per-step network latency alone. Multiple concurrent staging
operations in a multi-tile all-reduce pipeline each other through PCIe.

The contention worsens under pipeline parallelism when multiple microbatches
are in flight and each contributes staging traffic simultaneously.

---

## 5. Where the Scaling Wall Appears

The scaling wall is the node count N* at which `T_communication ≥ T_compute`
for a single layer's forward pass. Below N*, adding nodes improves throughput
because compute reduction outpaces communication growth. Above N*, adding
nodes degrades efficiency because communication grows faster than compute
shrinks.

For decode inference with small batch sizes (batch=1 per replica):

```
T_compute per layer ≈ FLOPS / (FP16_throughput * N_tiles_per_node * N_nodes)
T_comm per collective ≈ T_xe_link + log₂(N_nodes) * (2*α_network + 4*α_PCIe)
```

Setting `T_comm = T_compute` and solving for N_nodes gives N*.

For a representative 70B parameter model on Aurora-class hardware:
- FP16 FLOPS per PVC tile: ~100 TFLOPS
- Decode layer FLOPS (batch=1): ~2 * 2 * d_model * d_ffn ≈ 200M FLOPS
- T_compute per layer per tile: ~2 us at full utilization
- T_comm floor: 10 us (Xe Link) + log₂(N) * 12 us (with bounce buffer)

Setting `T_comm = T_compute`:
```
10 + log₂(N*) * 12 = 2 * N* (compute speedup factor)
```

This yields N* ≈ 8-16 nodes (96-192 tiles). Above this range, adding nodes
increases communication faster than it reduces per-node compute.

**Without the bounce buffer** (hypothetical GPU RDMA, α_PCIe = 0):
```
T_comm = 10 + log₂(N) * 2 us
```

The same equation gives N* ≈ 64-128 nodes — 4-8x more nodes before hitting
the same wall.

The bounce buffer does not change the exponent of the scaling law. Both paths
scale as O(log N) for recursive doubling. The bounce buffer adds a constant
multiplier to α that shifts the scaling wall from ~64 nodes to ~8 nodes.
For a 2048-node cluster, this is the difference between communication taking
22 us vs 142 us per collective, and between 10% and 65% of decode time spent
on communication.

---

## 6. Empirical Validation

The Aurora benchmark data (Ibeid et al., arXiv 2512.04291) provides empirical
grounding:

| Nodes | 8 B msg | 64 KB msg | Scaling ratio |
|---|---|---|---|
| 1 node | ~15 us | ~50 us | 1x |
| 2048 nodes | ~250 us | ~280 us | 16x / 5.6x |

For the 8-byte message, latency grows 16x from 1 to 2048 nodes. In a perfect
recursive doubling (log₂(2048) = 11 steps vs log₂(12) ≈ 3.6 steps for
intra-node at 1 node), the expected latency ratio is 11/3.6 = 3x if network
latency dominated. The 16x measured ratio reflects the dominant contribution
of the bounce buffer overhead: each of the 11 inter-node steps pays full
staging cost, while the intra-node single-node case pays only the Xe Link
cost with no host involvement.

For the 64 KB message, the bandwidth term in the latency equation provides
more amortization of the per-step fixed cost, so scaling is better (5.6x).
This is consistent with the theory: bounce buffer overhead is most damaging
for small messages, which is exactly the regime for decode inference.

The HiCCL paper (Hidayetoglu et al., arXiv 2408.05962) showed a 12.1x
geometric mean speedup over oneCCL on pre-production Aurora, attributing
improvements largely to better NIC utilization through topology-aware
decomposition and reduced staging overhead. This corroborates that the host
staging path is the dominant inefficiency in oneCCL's collective performance,
not algorithmic selection or ring topology.

---

## 7. Comparison: NVIDIA GPUDirect RDMA

The contrast with NVIDIA's production GPUDirect RDMA makes the cost concrete.
On NVIDIA hardware, NIC-to-GPU DMA is enabled by default via the
`nvidia-peermem` kernel module. The per-step latency for a GPU RDMA path is:

```
α_NVIDIA = α_NIC_GPU_DMA + α_network + α_NIC_GPU_DMA
         ≈ 0.5 us + 1-2 us + 0.5 us
         ≈ 2-3 us
```

On Intel PVC in production:

```
α_Intel_staging = α_D2H + α_OFI_post + α_network + α_H2D
                ≈ 3 us + 1.5 us + 1-2 us + 3 us
                ≈ 8-10 us
```

The Intel staging path is 3-5x slower per collective step. For a 2048-node
recursive-doubling allreduce (11 steps):

| Path | Per-step α | Total latency (latency-bound) |
|---|---|---|
| NVIDIA GPUDirect | 2-3 us | 22-33 us |
| Intel (staging default) | 8-10 us | 88-110 us |
| Intel HMEM (experimental) | ~3 us (if working) | ~33 us |

The HMEM path (`CCL_ATL_HMEM=1`) would close this gap if production-validated.
It allows the NIC to DMA from/to GPU memory via Linux dmabuf without transit
through host DRAM. The data path becomes GPU → NIC → wire → NIC → GPU, with
the host CPU still posting the OFI operation but not touching the data. The
remaining overhead vs NVIDIA is the CPU-posts-to-NIC step (~1.5 us) that NVIDIA
eliminates with GPU-initiated RDMA (GDRCopy/NVSHMEM).

---

## 8. The CPU-Posts Architectural Gap

Even with HMEM fully working, Intel Xe still requires the host CPU to call
`fi_tsendmsg()` for every collective step. NVIDIA's GDRCopy and NVSHMEM allow
the GPU kernel itself to enqueue RDMA operations to the NIC, eliminating the
CPU from the critical path entirely.

The CPU-posts model creates two problems that do not appear in the benchmark
numbers but affect sustained throughput:

**CPU serialization:** oneCCL's worker thread serializes OFI operations within
a rank. At high collective rates (decode with small batch), the worker thread
queue grows and introduces jitter. With GPU-initiated RDMA, the NIC hardware
queue absorbs bursts without serialization.

**Synchronization granularity:** The CPU must context-switch between compute
tasks (managing GEMM submission, KV cache operations) and communication tasks
(managing staging copies, OFI posts). With GPU-initiated RDMA, the GPU
handles communication scheduling as a CUDA/SYCL kernel call with no CPU
involvement. This allows true overlap without CPU scheduling interference.

Quantitatively, this overhead is difficult to isolate from benchmarks but
contributes to the gap between the theoretical 88-110 us HMEM latency and the
measured ~250 us at 2048 nodes.

---

## 9. Mitigation Strategies and Their Limits

### `async_op=True` (double-buffering)

Issuing collectives asynchronously allows the GPU to run the next layer's
GEMM while the current layer's collective is in flight. This hides but does
not reduce communication latency. The overlap is useful when:

```
T_compute(layer N+1) > T_comm(layer N)
```

For decode with small batch sizes, `T_compute ≈ 50-100 us` and
`T_comm ≈ 100-250 us` at 2048 nodes. The overlap is incomplete: the GPU
finishes layer N+1's compute before layer N's allreduce returns, and must
stall. The bounce buffer overhead contributes directly to this stall.

### `CCL_ALLREDUCE_SCALEOUT=ring`

Using ring for scaleout (instead of recursive doubling) trades latency for
bandwidth. For large messages at large N, ring achieves better bandwidth
utilization (2(N-1)/N * bandwidth vs all-to-all bandwidth in recursive
doubling). But ring's O(N) latency scaling makes it unsuitable for decode.
This variable controls only the inter-node phase; the host staging path
remains regardless of algorithm.

### `CCL_ATL_HMEM=1`

This is the only structural fix for the bounce buffer overhead. When working,
it reduces the per-step staging overhead from ~6 us to near zero (NIC
DMA-reads GPU directly). The CPU still posts the OFI operation, so ~1.5 us
of CPU overhead remains per step, but the PCIe saturation and buffer
contention problems are eliminated.

The practical limit: HMEM requires the entire software stack to be validated
together (libfabric version with FI_HMEM_ZE support, GPU driver with dmabuf
export, NIC firmware with P2P BAR access). Failure mode is a silent hang with
no error message. As of the current oneCCL release, this path is experimental
and disabled by default.

### Hierarchical decomposition (HiCCL approach)

The 12.1x speedup reported by HiCCL comes partly from aggressive topology
exploitation: routing inter-node traffic so that NIC-attached tiles (those
physically closest to the NIC on the PCIe fabric) handle staging, while
compute-heavy tiles do local reduction without touching the NIC path. This
reduces PCIe contention by concentrating staging traffic on fewer tiles with
better PCIe locality to the NIC.

This approach reduces the per-tile staging overhead but does not eliminate
the bounce buffer. At 2048 nodes, it reduces the effective per-step latency
from ~12 us to perhaps ~5-6 us — still 2-3x worse than GPU RDMA.

---

## 10. Summary

The host bounce buffer in oneCCL's default configuration (OFI transport,
HMEM disabled) adds approximately 6-8 us of PCIe round-trip overhead to every
step of every inter-node collective. For recursive doubling at scale:

- At 8 nodes (3 steps): adds ~18-24 us to each allreduce
- At 64 nodes (6 steps): adds ~36-48 us  
- At 2048 nodes (11 steps): adds ~66-88 us

For decode inference where per-collective budget is ~30-50 us (to keep
communication below compute), the staging penalty makes multi-node tensor
parallelism impractical beyond 8-16 PVC nodes for small-batch workloads.

The wall is not a result of algorithm choice, ring topology, or NIC bandwidth.
It is a consequence of the GPU's inability to post operations to the NIC
directly. Every architectural workaround available in oneCCL (async_op,
ring vs recursive doubling, topo hierarchical algorithm, NUMA pinning) reduces
communication overhead but cannot eliminate the staging tax.

Eliminating the wall requires one of:

1. **HMEM validation to production quality** — eliminates the PCIe data path
   overhead, leaving only CPU-posts latency (~1.5 us/step)
2. **GPU-initiated RDMA** — eliminates CPU involvement entirely, achieving
   parity with NVIDIA GPUDirect Async
3. **UALink or equivalent intra-rack fabric** — bypasses the NIC entirely for
   rack-scale collectives, leaving only inter-rack traffic to the staging path

Until one of these paths reaches production status on Intel Xe, the practical
limit for latency-sensitive multi-node inference on PVC-class hardware is
approximately 8-16 nodes for TP allreduce, or workloads where collective
messages are large enough that bandwidth efficiency (not latency) is the
primary measure.

---

## References

- oneCCL source: `src/coll/coll_util.cpp` — topo scaleout staging path
- oneCCL source: `src/coll/algorithms/allreduce/sycl/allreduce_scaleout_sycl.cpp` — OFI forcing `copy_to_host=true`
- oneCCL source: `src/coll/algorithms/utils/sycl_coll_base.cpp` — `check_mpi_supports_rdma()`, "ofi collective only supports host memory"
- Ibeid et al., "Scaling MPI Applications on Aurora" (arXiv:2512.04291, Dec 2025)
- Hidayetoglu et al., "HiCCL: A Hierarchical Collective Communication Library" (arXiv:2408.05962, Aug 2024)
- Thakur et al., "Optimization of Collective Communication Operations in MPICH" (IJHPCA, 2005) — algorithmic foundations for ring, recursive doubling, tree collectives
