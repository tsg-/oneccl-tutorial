# GPU Hardware Constraints

This chapter documents the hardware-level constraints that shape how oneCCL
implements collectives on CRI (Crescent Island, Xe3) — and provides historical
context from PVC (Ponte Vecchio, Xe-HPC) which introduced similar challenges.

---

## The Two Constraints That Matter

| Constraint | PVC (Xe-HPC, 2022) | CRI (Xe3, 2026) |
|---|---|---|
| Intra-node GPU fabric | Xe Link (high-BW coherent) | **None — PCIe only** |
| GPU Direct DMA to NIC | No | No |

**CRI is more constrained than PVC.** PVC had Xe Link for fast intra-node GPU-to-GPU
transfers (used on systems like Aurora with 6 GPUs per node). CRI has no GPU fabric
at all — every GPU-to-GPU transfer, even within the same node, traverses PCIe and
potentially UPI.

Both generations lack **GPU Direct DMA** — the GPU cannot initiate network transfers.
All inter-node communication requires host CPU staging.

---

## GPU Direct DMA: What Neither Generation Has

**GPU Direct DMA** (also called GPU-initiated communication or GPU RDMA) allows a GPU
kernel to directly post network operations (RDMA sends/receives) without involving the
host CPU. NVIDIA's GPUDirect RDMA + CUDA kernels can do this. Intel Xe GPUs cannot.

On CRI, the GPU has no path to initiate network I/O:

```
                  ┌─── CRI GPU (Xe3) ───┐
                  │  Compute (EUs)       │
                  │  Copy Engines        │──── Level Zero IPC (intra-node P2P via PCIe)
                  │  L1/L2 Cache         │
                  └──────────┬───────────┘
                             │ PCIe
                             ▼
                  ┌─── Host CPU ───┐
                  │  oneCCL worker  │──── NIC (OFI/verbs) ──── Network
                  │  staging buffer │
                  └────────────────┘

  GPU cannot talk to the NIC directly.
  All inter-node traffic goes through host staging.
```

This means:

1. **Every scaleout collective requires a GPU→host copy before network transfer**
2. **Every received message requires a host→GPU copy after network transfer**
3. The host CPU is always on the critical path for inter-node communication

---

## No Fabric: How CRI Differs from PVC

PVC systems (e.g., Aurora) use Xe Link to form a high-bandwidth mesh between GPUs
within a node. The `topo` algorithm's scale-up phase exploits this:

```
PVC with Xe Link (Aurora-class):
  GPU0 ←──Xe Link──→ GPU1    (direct, high-BW, coherent)
  Scale-up: ~100+ GB/s per link, no host involvement

CRI without fabric:
  GPU0 ←──PCIe──→ CPU ←──PCIe──→ GPU1    (host-mediated)
  Scale-up: limited by PCIe BW (~32 GB/s Gen5 x16), crosses CPU
```

On CRI, even the **scale-up phase** (intra-node) is constrained:
- No direct GPU-to-GPU path exists
- Level Zero IPC handles enable P2P DMA over PCIe, but bandwidth is limited
- Cross-socket GPU pairs traverse UPI, adding latency and contention
- This is why NUMA pinning is the single biggest performance lever on CRI

---

## How oneCCL Handles This: The `topo` Algorithm

The `topo` algorithm is oneCCL's solution for GPU hardware without direct network access.
It splits every collective into phases:

```
Scale-up phase (intra-node):
  GPU ←→ GPU via Level Zero IPC handles (PCIe P2P DMA on CRI)
  Uses copy engines for P2P transfers
  No host staging for same-node GPUs (when P2P access works)

Scaleout phase (inter-node):
  GPU → host staging buffer (PCIe DMA)
  Host → NIC → network → remote host (OFI transport)
  Remote host → remote GPU (PCIe DMA)
```

This is why `CCL_ALLREDUCE=topo` (the default) must remain set for GPU buffers.
Setting any other value (e.g., `ring`) tells oneCCL to copy the **entire** buffer
to host memory and run a CPU-side algorithm — losing even the PCIe P2P scale-up path.

---

## Copy Engines vs Compute Kernels

CRI has dedicated **copy engines** separate from the compute EUs:

| Engine | What It Does | When Used |
|---|---|---|
| Main copy engine | General DMA: GPU↔host, GPU↔GPU (P2P over PCIe) | Default for all transfers |
| Link copy engine | Dedicated to P2P over fabric links | Only with Xe Link (not on CRI) |
| Compute EUs | GEMM, attention, etc. | Not used for communication |

On CRI (no Xe Link), only the main copy engine is available for P2P. The link
copy engine has no fabric to drive.

Relevant variables:

```bash
# Which copy engine to use (CRI: only 'main' is useful)
export CCL_ZE_COPY_ENGINE=main

# Monolithic pipeline kernel: fuses reduce-scatter into a single kernel submission
# Reduces command queue overhead for small messages
export CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL=1
```

The copy engine operates concurrently with compute EUs. This means oneCCL can
overlap the PCIe DMA (for the next layer's collective) with the current layer's
GEMM — but only if collectives are issued asynchronously (see `async_op=True`
pattern in [notebook 04](../notebooks/04_inference_tp_decode)).

---

## P2P Access and Atomics

oneCCL's topology manager probes P2P capabilities at init time:

```
check_p2p_access():
  For each GPU pair, test if Level Zero IPC memory handles work.
  On CRI: P2P works within a node via PCIe (no fabric).
  Cross-node: no P2P — must use host staging.

check_p2p_atomics():
  Test if GPU atomic operations work across devices.
  Xe GPUs: atomics across devices may not work reliably over PCIe.
  oneCCL falls back to non-atomic paths when this check fails.
```

If P2P access fails (e.g., due to IOMMU misconfiguration or missing
`/dev/dri/renderD*` permissions), oneCCL falls back entirely to host-staged
communication even for intra-node transfers — a significant performance loss.

**Verify P2P is working:**
```bash
# Check that Level Zero sees all GPUs and can open IPC handles
CCL_LOG_LEVEL=info mpirun -n 4 python -c "
import torch.distributed as dist
import oneccl_bindings_for_pytorch
dist.init_process_group(backend='ccl')
dist.destroy_process_group()
" 2>&1 | grep -i "p2p\|ipc\|access"
```

---

## GPU Memory Registration (dmabuf) — Experimental

Linux 5.12+ provides **dmabuf** (DMA buffer sharing) which allows a NIC to
directly access GPU memory regions without host staging:

```
Without dmabuf (current default):
  GPU buf → copy to host → NIC reads from host → network

With dmabuf (experimental):
  GPU buf → NIC reads directly via dmabuf fd → network
  (eliminates one host copy on the send path)
```

To enable on supported configurations:

```bash
export CCL_ATL_HMEM=1       # Enable heterogeneous memory support
export FI_MR_CACHE_MONITOR=userfaultfd  # Required for OFI memory registration
```

**Requirements:**
- Linux kernel ≥ 5.12 with CONFIG_DMABUF enabled
- Intel GPU driver with dmabuf export support
- OFI provider that supports FI_HMEM (e.g., verbs with peer-memory or PSM3)

This is **not production-ready** on current CRI deployments but represents the
path toward eliminating host staging on the send side.

---

## TMP_BUF: Async Collectives via Staging

For workloads that need non-blocking collective semantics (fire-and-forget with
later synchronization), oneCCL provides `TMP_BUF` mode:

```bash
export CCL_ALLREDUCE__TMP_BUF=1
```

What this does:
1. oneCCL copies the input to an internal staging buffer immediately
2. The user's buffer is free for reuse without waiting
3. oneCCL runs the collective on the staging buffer in the background
4. On completion, oneCCL copies the result back to the output buffer

**Tradeoff:** Adds two extra copies (in + out) but makes the collective fully
asynchronous from the user's perspective. Useful when:
- You need to overlap collective with compute on the **same** buffer
- Latency tolerance is high but throughput matters (prefill, not decode)
- You cannot restructure code to use double-buffering

For decode (latency-critical), the extra copies usually hurt more than the
overlap helps. Double-buffering with `async_op=True` is preferred.

---

## What Changes with Future Hardware (UALink)

| Constraint | CRI (Xe3) | Future (UALink-equipped) |
|---|---|---|
| GPU Direct DMA (GPU-initiated NIC) | No | Expected (HW RDMA engine) |
| Intra-node fabric | None (PCIe only) | UALink (high-BW open mesh) |
| P2P atomics | Unreliable over PCIe | Full support expected |
| Copy engines for fabric | Main only (no link target) | Link engines + UALink |
| NIC integration | Discrete (PCIe-attached) | Closer integration expected |
| Host staging required? | Yes (scaleout) | No (with GPU RDMA) |

When GPU Direct DMA and UALink arrive:
- Intra-node scale-up uses fabric instead of PCIe — bandwidth jumps dramatically
- Scaleout collectives eliminate the host→NIC→host copy
- `topo` algorithm can pipeline GPU-initiated sends with compute
- Latency drops by 1-2 PCIe round-trips per collective (~2-5 µs each)
- `offload` mode for send/recv becomes viable (currently requires HW RDMA)

---

## Practical Implications for CRI Deployment

Given CRI's constraints (no fabric, no GPU Direct DMA):

1. **Never set `CCL_ALLREDUCE=ring` (or any non-topo value) for GPU inference.**
   This forces GPU→host→CPU-algorithm→host→GPU for every collective. Use
   `CCL_ALLREDUCE_SCALEOUT=ring` to control inter-node only.

2. **Verify P2P access works.** Without it, even intra-node communication falls
   back to host staging. Check `/dev/dri/` permissions and IOMMU settings.

3. **Pin ranks to match GPU NUMA affinity.** The `topo` algorithm's scale-up phase
   uses P2P DMA over PCIe — misaligned pinning routes DMA through the wrong PCIe
   root complex, crossing UPI unnecessarily.

4. **Accept the host staging tax for scaleout.** There is no way to eliminate the
   host copy for inter-node communication. Budget ~2-5 µs per collective for the
   extra PCIe round-trip.

5. **Use `async_op=True` to overlap.** While one collective is in the host staging
   phase, the GPU can run compute for the next layer. This is the primary
   mechanism to hide communication latency.

6. **Don't enable `CCL_ATL_HMEM=1` in production** unless explicitly validated on
   your driver/kernel/NIC stack. It's experimental and can cause silent hangs.

See [Perf Tuning](perf_tuning) for the full variable reference and launch templates.

---

## Published Benchmarks: Collective Performance at Scale

### Aurora Allreduce Latency (PVC + Xe Link + Slingshot-11)

From Ibeid et al., "Scaling MPI Applications on Aurora" (arXiv:2512.04291):

| Message Size | 1 Node (6 GPUs) | 2048 Nodes (~12K GPUs) |
|---|---|---|
| 8 B | ~15 µs | ~250 µs |
| 512 B | ~20 µs | ~250 µs |
| 2 KB | ~25 µs | ~260 µs |
| 64 KB | ~50 µs | ~280 µs |

Latency growth is sub-linear (recursive-doubling/tree algorithms). Aurora uses 8×
Slingshot-11 NICs per node (200 Gbps each), with ~23-25 GB/s per NIC for GPU buffers.

### oneCCL vs Optimized Libraries (Pre-Production Aurora, 2024)

From Hidayetoglu et al., "HiCCL: A Hierarchical Collective Communication Library"
(arXiv:2408.05962), tested on 4 Aurora nodes (48 GPU tiles):

| Collective | oneCCL Throughput | HiCCL Throughput | Gap |
|---|---|---|---|
| Allreduce | ~40-80 GB/s | ~80-120 GB/s | 1.5-2× |
| Broadcast | ~20-40 GB/s | ~80-120 GB/s | 2-4× |
| All-to-All | ~20-40 GB/s | ~40-80 GB/s | 2× |

HiCCL reported a **12.1× geometric mean improvement** over oneCCL across all
collectives. Important caveats:
- This was pre-production Aurora with early oneCCL (2024)
- oneCCL's multi-NIC striping and hierarchical algorithms were not yet tuned for
  Aurora's 12-GPU, 8-NIC topology
- Production oneCCL has improved significantly since

### Large-Scale Training Efficiency

From Vooturi et al., "Scalable Pretraining of Large MoE Language Models on Aurora"
(arXiv:2604.00785):
- 220B parameter MoE model scaled from 384 to 12,288 PVC GPU tiles
- **~90% scaling efficiency** at 12,288 tiles
- Communication overhead managed via expert-parallel sharding

### What This Means for CRI

CRI nodes are more constrained than Aurora:
- No Xe Link (Aurora has 28 GB/s per Xe Link between GPU stacks)
- Fewer NICs per node (Aurora has 8× Slingshot-11)
- PCIe-only intra-node path

Expect **higher per-collective latency** and **lower bandwidth** than the Aurora numbers
above. The Aurora benchmarks represent an upper bound for what Intel GPU collective
communication achieves with fabric assistance. CRI without fabric will be closer to
the small-message latency floor (~15-25 µs per collective) but will not scale bandwidth
as aggressively with message size.

---

## References

### Hardware & Source
- [Intel Data Center GPU Max Series (PVC) Product Specs](https://www.intel.com/content/www/us/en/products/sku/232873/intel-data-center-gpu-max-1550/specifications.html) — Xe Link frequency, PCIe Gen5, tile architecture
- [oneCCL Source: `topo_manager.cpp`](https://github.com/oneapi-src/oneCCL/blob/master/src/topology/topo_manager.cpp) — `check_p2p_access()`, `check_p2p_atomics()`, fabric connectivity probing
- [oneCCL Source: `ze_primitives.cpp`](https://github.com/oneapi-src/oneCCL/blob/master/src/sched/entry/ze/ze_primitives.cpp) — `device_family` enum, `should_disable_rdma()`, copy engine selection
- [oneCCL Documentation: Environment Variables](https://github.com/oneapi-src/oneCCL/blob/master/doc/rst/source/env-variables.rst) — `CCL_ATL_HMEM`, `CCL_ZE_COPY_ENGINE`, `TMP_BUF`, `offload` mode
- [oneCCL Documentation: dmabuf Support](https://github.com/oneapi-src/oneCCL/blob/master/doc/rst/source/advanced-configuration/dmabuf.rst) — GPU memory registration via Linux dmabuf/OFI verbs
- [UALink Consortium](https://ualink.org/) — open standard for accelerator interconnects (Intel founding member)

### Benchmarks & Papers
- Ibeid et al., "Scaling MPI Applications on Aurora" (arXiv:2512.04291, Dec 2025) — MPI collective latency/bandwidth at 2048+ nodes on PVC
- Hidayetoglu et al., "HiCCL: A Hierarchical Collective Communication Library" (arXiv:2408.05962, Aug 2024) — oneCCL vs optimized collectives on pre-production Aurora
- Vooturi et al., "Scalable Pretraining of Large MoE Language Models on Aurora" (arXiv:2604.00785, Apr 2026) — 90% scaling efficiency at 12,288 PVC tiles
- Ma et al., "CoCoDiff: Optimizing Collective Communications for Distributed Diffusion Transformer Inference" (arXiv:2604.14561, Apr 2026) — 3.6× avg speedup for all-to-all on Aurora via topology-aware decomposition
