# GPU Hardware Constraints

The previous chapters established what collectives do ([Foundations](00_foundations)),
how oneCCL implements them ([Overview](01_overview)), and why ring construction matters
on NUMA hardware ([Topology](02_topology)). This chapter explains the physical hardware
limits that govern all of those choices on BMG/CRI — specifically, *why* certain algorithm
shortcuts that work on NVIDIA hardware do not work here.

**What you need to understand for the Hands-On notebooks:** The two constraints at the
top of this chapter (no intra-node GPU fabric, no GPU-initiated network I/O). Everything
past the `topo` algorithm section is a deep dive into HMEM internals and scaling analysis
that you can return to when diagnosing production performance issues.

This chapter covers BMG/CRI (Battlemage / Crescent Island, Xe3) with context from PVC
(Ponte Vecchio, Xe-HPC). PVC and BMG/CRI both lack GPU-initiated network I/O, but
differ sharply on intra-node fabric: PVC has Xe Link, BMG/CRI has only PCIe.

---

## The Two Constraints that Matter

| Constraint | PVC (Xe-HPC, 2022) | BMG/CRI (Xe3, 2026) |
|---|---|---|
| Intra-node GPU fabric | Xe Link (high-BW coherent) | **None, PCIe only** |
| GPU-initiated network I/O | No | No |
| NIC-to-GPU DMA (HMEM/dmabuf) | Experimental (`CCL_ATL_HMEM=1`) | Experimental (`CCL_ATL_HMEM=1`) |
| `offload` mode (Xe Link RDMA) | Yes (intra-node only) | No (no fabric) |
| **Scaleout default** | **Host-staged** | **Host-staged** |

**Host staging** means: before a GPU can send data to another node over the
network, the data must first be copied from GPU memory to a CPU-side buffer,
then the CPU posts the send to the NIC. On receipt, the NIC writes into a CPU
buffer, and the CPU copies to GPU memory. Every inter-node collective goes
through this two-copy path by default.

Neither PVC nor BMG/CRI can bypass this: the GPU cannot autonomously post network
operations. The host CPU always orchestrates every send/recv to the NIC. In
standard deployments, all inter-node (scaleout) traffic is copied through host
memory. By comparison, NVIDIA GPUDirect RDMA has been production-grade for over a decade and
eliminates this path for CUDA workloads.

The term "GPU Direct" is often used loosely. There are actually four separate
capabilities, each with different hardware requirements:

| Capability | NVIDIA | PVC (Xe-HPC) | BMG/CRI (Xe3) |
|---|---|---|---|
| **GPU-initiated network I/O** | Yes (GPUDirect Async) | No | No |
| **NIC-to-GPU DMA (host-orchestrated)** | Yes (GPUDirect RDMA) | Experimental (HMEM) | Experimental (HMEM) |
| **GPU-to-GPU P2P DMA** | Yes (NVLink / PCIe) | Yes (Xe Link + PCIe) | PCIe only |
| **GPU RDMA offload** | Yes | Xe Link only (`offload` mode) | No (no fabric) |

On BMG/CRI, the GPU has no path to initiate network I/O:

```
  ┌──────────────────────┐
  │    BMG/CRI GPU (Xe3)     │
  ├──────────────────────┤
  │  Compute (EUs)       │
  │  Copy Engines        │──── Level Zero IPC (GPU driver P2P over PCIe)
  │  L1/L2 Cache         │
  └───────────┬──────────┘
              │ PCIe
              ▼
  ┌──────────────────────┐
  │      Host CPU        │
  ├──────────────────────┤
  │  oneCCL worker       │──── NIC (OFI/verbs) ──── Network
  │  staging buffer      │
  └──────────────────────┘

  GPU cannot talk to the NIC directly.
  Default: all inter-node data transits host memory (host staging).
  With HMEM (experimental): NIC reads/writes GPU memory directly.
```

---

## BMG/CRI vs PVC: Absence of Intra-Node Fabric

A collective running across multiple nodes has two communication phases:
**scale-up** (within a node, GPU-to-GPU) and **scaleout** (between nodes,
through NICs). The scale-up phase determines how fast GPUs on the same server
can exchange partial results. The scaleout phase determines how fast those
results move across the network.

PVC systems (e.g., Aurora, the DOE supercomputer at Argonne) used Xe Link to
form a high-bandwidth mesh between GPUs within a node. BMG/CRI has no equivalent:

```
PVC with Xe Link (Aurora-class):
  GPU0 ←──Xe Link──→ GPU1    (direct, high-BW, coherent)
  Scale-up: ~100+ GB/s per link, no host involvement

BMG/CRI without fabric:
  GPU0 ←──PCIe──→ CPU ←──PCIe──→ GPU1    (host-staged)
  Scale-up: limited by PCIe BW (~32 GB/s Gen5 x16), crosses CPU
```

On BMG/CRI, even the **scale-up phase** (intra-node) is constrained:
- No direct GPU-to-GPU path exists without going through PCIe
- Level Zero IPC handles enable P2P DMA over PCIe, but bandwidth is limited
- Cross-socket GPU pairs traverse UPI, adding latency and contention
- NUMA pinning is the biggest performance lever on BMG/CRI for this reason

---

## The topo Algorithm: Host-Staged Hierarchical Communication

The `topo` algorithm is oneCCL's default for GPU buffers. It exploits the two-level
topology (intra-node fast, inter-node slower) by splitting every collective into phases.
On BMG/CRI, Level Zero IPC handles are the mechanism for intra-node P2P: each GPU
exports a memory handle via the Level Zero driver, and a peer GPU imports it to
do a direct DMA read/write over PCIe without going through the host.

```
Scale-up phase (intra-node, same server):
  GPU ←→ GPU via Level Zero IPC handles (PCIe P2P DMA on BMG/CRI)
  Uses copy engines; no CPU involvement when P2P access works

Scaleout phase (inter-node, different servers):
  GPU → host staging buffer (PCIe DMA, ~2 us)
  Host → NIC → network → remote host (OFI/libfabric, ~5 us)
  Remote host → remote GPU (PCIe DMA, ~2 us)
```

For a 2-node allreduce with TP=4 per node, the full data flow:

```
  Node 0                              Node 1
  ┌───────────────────────────┐       ┌───────────────────────────┐
  │ GPU0  GPU1  GPU2  GPU3    │       │ GPU4  GPU5  GPU6  GPU7    │
  │   │     │     │     │     │       │   │     │     │     │     │
  │   └─────┴─────┴─────┘     │       │   └─────┴─────┴─────┘     │
  │   Scale-up: PCIe P2P      │       │   Scale-up: PCIe P2P      │
  │   (reduce-scatter local)  │       │   (reduce-scatter local)  │
  │         │                 │       │         │                 │
  │         ▼                 │       │         ▼                 │
  │   ┌──────────┐            │       │   ┌──────────┐            │
  │   │ staging  │            │       │   │ staging  │            │
  │   │  buffer  │            │       │   │  buffer  │            │
  │   └─────┬────┘            │       │   └─────┬────┘            │
  │         │                 │       │         │                 │
  │        NIC ───────────────┼─ OFI ─┼──────  NIC                │
  │         │                 │       │         │                 │
  │         ▼                 │       │         ▼                 │
  │   Allgather local         │       │   Allgather local         │
  │   ┌─────┬─────┬─────┐     │       │   ┌─────┬─────┬─────┐     │
  │   │     │     │     │     │       │   │     │     │     │     │
  │ GPU0  GPU1  GPU2  GPU3    │       │ GPU4  GPU5  GPU6  GPU7    │
  └───────────────────────────┘       └───────────────────────────┘
```

`CCL_ALLREDUCE=topo` (the default) must remain set for GPU buffers.
Setting any other value (e.g., `ring`) tells oneCCL to copy the **entire** buffer
to host memory and run a CPU-side algorithm, losing even the PCIe P2P scale-up path.

---

## Quick Deployment Takeaways

If you are configuring a deployment and not debugging HMEM internals:

1. Leave `CCL_ALLREDUCE` unset (defaults to `topo`). Setting it to anything else drops to host-staged CPU algorithms.
2. Inter-node traffic is host-staged by default on both PVC and BMG/CRI. This is the current verified baseline.
3. `CCL_ATL_HMEM=1` can bypass host staging if the libfabric build supports `FI_HMEM_ZE` — verify before relying on it (see [Host Staging §3](host_staging_scaling)).
4. For decode (small messages): leave scaleout algorithm on `auto`/`direct`. For training (large messages): set `CCL_ALLREDUCE_SCALEOUT=ring`.
5. Pin ranks to sockets (`I_MPI_PIN_DOMAIN=socket`) to keep intra-node ring steps on-socket.

The sections below explain the hardware constraints behind these recommendations.

---

## Scaleout Limitation: Host Staging Is the Default

Both PVC and BMG/CRI host-stage inter-node traffic by default. This is the
per-step overhead described in [Host Staging](host_staging_scaling).

OFI (OpenFabrics Interfaces) is the network transport that oneCCL uses for
inter-node communication — it's the layer between oneCCL and the physical NIC
(InfiniBand, Slingshot, or Ethernet with RDMA). By default, OFI expects data
in host memory. Getting it to work directly with GPU memory requires HMEM
support, described later.

oneCCL has two algorithm families, and both default to host staging:

**1. The `topo` algorithm (default for GPU buffers):**

The scaleout phase checks a flag called `enable_hmem`. If it's not set (the default),
it copies the tensor to a host-side staging buffer before calling the network transport.
In `coll_util.cpp`:
```
if (!enable_hmem) {
    LOG_DEBUG("topo/scale_out: use host_...");
    // D2H copy into staging buffer
}
```

**2. SYCL scaleout kernels (used for the actual collective on the staged data):**

These kernels have a separate gate: even if you set `CCL_SYCL_ENABLE_DIRECT_GPU_RDMA=1`,
the code overrides it to `copy_to_host = true` whenever the transport is OFI.
From `allreduce_scaleout_sycl.cpp`:
```cpp
bool copy_to_host = sycl_enable_direct_gpu_rdma ? false : true;
if (should_disable_rdma(ze_dev) || atl_transport == ccl_atl_ofi) {
    copy_to_host = true;  // OFI always host-stages
}
```

The comment in `sycl_coll_base.cpp` at `check_mpi_supports_rdma()` states:
`"ofi collective only supports host memory"`.

**Why `CCL_SYCL_ENABLE_DIRECT_GPU_RDMA` doesn't work with OFI:** This knob controls
the SYCL scaleout code path in `allreduce_scaleout_sycl.cpp`, which calls
`atl_comm->allreduce()` — OFI's collective-level interface that expects host-accessible
buffers. `CCL_ATL_HMEM=1` is a separate mechanism: it enables OFI-level HMEM
registration (`fi_mr_regattr` with `FI_HMEM_ZE`) on the point-to-point send/recv path,
allowing the NIC to DMA from GPU memory directly. The two are independent code paths
gated at different levels (see the split-path explanation in
[Host Staging](host_staging_scaling) §2.0).

**Result:** `CCL_SYCL_ENABLE_DIRECT_GPU_RDMA=1` only works with **MPI transport**
(requires `I_MPI_OFFLOAD=2` with Intel MPI, or MPICH with
`MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1`). OFI transport, which is the recommended
transport for inference, cannot use this path.

The full decision logic, from transport choice down to whether the NIC ever
touches GPU memory directly:

```
Scaleout Data Path Decision (for GPU buffers, inter-node):

  Is CCL_ATL_TRANSPORT=ofi?
  │
  ├── YES (default, recommended for inference)
  │     │
  │     ├── Is CCL_ATL_HMEM=1?
  │     │     │
  │     │     ├── YES → NIC DMA from/to GPU (experimental, may hang)
  │     │     │         Data: GPU ──NIC──▶ network ──▶ NIC──GPU
  │     │     │
  │     │     └── NO (default) → Host staging
  │     │                        Data: GPU→Host→NIC→net→NIC→Host→GPU
  │     │
  │     └── CCL_SYCL_ENABLE_DIRECT_GPU_RDMA? → IGNORED (OFI can't use it)
  │
  └── NO (CCL_ATL_TRANSPORT=mpi)
        │
        ├── Is CCL_SYCL_ENABLE_DIRECT_GPU_RDMA=1 + I_MPI_OFFLOAD=2?
        │     │
        │     ├── YES → MPI handles GPU RDMA (via Intel MPI offload)
        │     │         Data: GPU ──MPI──▶ network ──▶ MPI──GPU
        │     │
        │     └── NO → Host staging via MPI
        │
        └── Is should_disable_rdma(device)? → Force host staging
              (only affects ARC B-series: 0xE20B-0xE223)
```

---

## Copy Engines and Compute Overlap

BMG/CRI has dedicated **copy engines** separate from the compute units (EUs —
execution units, the shader cores that run GEMM and attention kernels):

| Engine | What It Does | When Used |
|---|---|---|
| Main copy engine | General DMA: GPU-host, GPU-GPU (P2P over PCIe) | Default for all transfers |
| Link copy engine | Dedicated to P2P over fabric links | Only with Xe Link (not on BMG/CRI) |
| Compute EUs | GEMM, attention, etc. | Not used for communication |

On BMG/CRI (no Xe Link), only the main copy engine is available for P2P. The link
copy engine has no fabric to drive.

Relevant variables:

```bash
# Which copy engine to use (BMG/CRI: only 'main' is useful)
export CCL_ZE_COPY_ENGINE=main

# Monolithic pipeline kernel: fuses reduce-scatter into a single kernel submission
# Reduces command queue overhead for small messages
export CCL_REDUCE_SCATTER_MONOLITHIC_PIPELINE_KERNEL=1
```

The copy engine operates concurrently with compute EUs. oneCCL can overlap the
PCIe DMA (for the next layer's collective) with the current layer's GEMM, but
only if collectives are issued asynchronously (see `async_op=True` pattern in
[notebook 04](../notebooks/04_inference_tp_decode)).

```
Timeline (async_op=True, double-buffered):

              Layer N          Layer N+1        Layer N+2
              ─────────────    ─────────────    ─────────────
Compute EUs:  ████ GEMM N ███  ████ GEMM N+1█  ████ GEMM N+2█
Copy Engine:     ░░░ allreduce N-1 ░░░░
                               ░░░ allreduce N ░░░░
                                                ░░░ allreduce N+1 ░░░░
              ├──────────────┤├──────────────┤├──────────────┤
                             ▲               ▲
                 compute and DMA overlap     no stalls, GPU always busy


Timeline (sync, no overlap, DEFAULT without async_op):

              Layer N       wait   Layer N+1     wait   Layer N+2
              ───────────── ────── ───────────── ────── ─────────
Compute EUs:  ████ GEMM N █        ████ GEMM N+1        ████ GEMM █
Copy Engine:               ░░ AR N ░░           ░░AR N+1░░
              ├─────────────┤├─────┤├────────────┤├─────┤
                            ▲ STALL ▲            ▲ STALL▲
                   GPU idle while allreduce completes
```

The async path hides communication latency behind compute. For decode (small GEMMs,
~50 us), the overlap window is tight: communication must complete within one layer's
compute time or it stalls the next layer.

---

## P2P Access

oneCCL's topology manager probes P2P capabilities at init time:

```
check_p2p_access():
  For each GPU pair, test if Level Zero IPC memory handles work.
  On BMG/CRI: P2P works within a node via PCIe (no fabric).
  Cross-node: no P2P, must use host staging.

check_p2p_atomics():
  Test if GPU atomic operations work across devices.
  Xe GPUs: atomics across devices may not work reliably over PCIe.
  oneCCL falls back to non-atomic paths when this check fails.
```

The result is a P2P connectivity matrix like:

```
P2P Connectivity Matrix (BMG/CRI 4-GPU node, 2 sockets):

         GPU0   GPU1   GPU2   GPU3
GPU0      -     PCIe   UPI    UPI     Socket 0: GPU0, GPU1
GPU1    PCIe     -     UPI    UPI     Socket 1: GPU2, GPU3
GPU2    UPI    UPI      -     PCIe
GPU3    UPI    UPI    PCIe     -

Legend:
  PCIe = same-socket P2P via PCIe root complex (~25-32 GB/s)
  UPI  = cross-socket P2P via UPI bridge (~10-15 GB/s under load)

If IOMMU or ACS blocks P2P:

         GPU0   GPU1   GPU2   GPU3
GPU0      -     HOST   HOST   HOST    All pairs fall back to
GPU1    HOST     -     HOST   HOST    host-staged copy (~5-10 GB/s)
GPU2    HOST   HOST     -     HOST
GPU3    HOST   HOST   HOST     -
```

If P2P access fails (e.g., due to IOMMU misconfiguration or missing
`/dev/dri/renderD*` permissions), oneCCL falls back entirely to host-staged
communication even for intra-node transfers, which costs 3-5x bandwidth.

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

## NIC-to-GPU DMA via HMEM (CCL_ATL_HMEM), Experimental

The decision tree above shows that the default path always host-stages. HMEM
is the escape hatch: a mechanism where the NIC reads and writes GPU memory
directly via PCIe, bypassing the host staging buffer entirely.

`CCL_ATL_HMEM=1` is the **only mechanism** for avoiding host staging with OFI
transport. When set, oneCCL registers GPU memory directly with the libfabric
transport layer using the `FI_HMEM` capability. The NIC performs RDMA
reads/writes from/to GPU memory, eliminating the staging copy on both send
and receive paths.

```
Without HMEM (current default):
  Send: GPU buf → copy to host staging → NIC reads host → network
  Recv: network → NIC writes host → copy to GPU buf

With HMEM enabled:
  Send: GPU buf → NIC reads GPU memory directly → network
  Recv: network → NIC writes GPU memory directly
  (host CPU still orchestrates, but data path skips host memory)
```

### Internal Mechanism

This subsection is for readers who want to understand what happens at the
libfabric and kernel level. It's not required for deployment; skip to
[No Hardware-Specific Gating](no-hardware-specific-gating) if you just want
to know whether BMG/CRI supports HMEM.

The mechanism in oneCCL (from `src/atl/ofi/atl_ofi.cpp`):

1. At init, oneCCL opens a libfabric provider with `FI_HMEM` capability and
   `FI_MR_HMEM` memory registration mode
2. On each send/recv, oneCCL calls `zeMemGetAllocProperties()` to determine if
   the buffer is GPU memory
3. For GPU buffers, it registers with `fi_mr_regattr()` using `iface=FI_HMEM_ZE`
   and the Level Zero device index
4. The registered MR descriptor is passed to `fi_tsendmsg()`/`fi_trecvmsg()`
5. The libfabric verbs provider internally uses Linux dmabuf (kernel 5.12+) to
   allow the NIC to DMA from/to the GPU BAR (the PCIe memory window through
   which the NIC can access GPU VRAM)

```
HMEM Registration and Transfer Flow:

  ┌───────────────────────────────────────────────────────────────┐
  │                      oneCCL (atl_ofi.cpp)                     │
  │                                                               │
  │  1. zeMemGetAllocProperties(buf)                              │
  │     → "this is ZE device memory on device idx N"              │
  │                                                               │
  │  2. fi_mr_regattr(buf, len, iface=FI_HMEM_ZE, device=N)       │
  │     → MR handle (cached for subsequent calls)                 │
  │                                                               │
  │  3. fi_tsendmsg(ep, msg, MR) / fi_trecvmsg(ep, msg, MR)       │
  │     → NIC uses MR to locate GPU BAR mapping                   │
  └─────────────────────────────┬─────────────────────────────────┘
                                │
  ┌─────────────────────────────▼─────────────────────────────────┐
  │                  libfabric (verbs provider)                   │
  │                                                               │
  │  fi_mr_regattr → ibv_reg_dmabuf_mr(dmabuf_fd, offset, len)    │
  │  fi_tsendmsg   → ibv_post_send(wr with GPU-mapped lkey)       │
  │  fi_trecvmsg   → ibv_post_recv(wr with GPU-mapped lkey)       │
  └─────────────────────────────┬─────────────────────────────────┘
                                │
  ┌─────────────────────────────▼─────────────────────────────────┐
  │                  NIC Hardware (RDMA engine)                   │
  │                                                               │
  │  DMA read from GPU BAR ──→ wire ──→ DMA write to GPU BAR      │
  │  (no host memory involved in data path)                       │
  └───────────────────────────────────────────────────────────────┘
```

oneCCL never passes dmabuf file descriptors directly. The dmabuf mechanism is
abstracted behind libfabric's `FI_HMEM` API.

(no-hardware-specific-gating)=
### No Hardware-Specific Gating

HMEM has **no device-family restrictions** in oneCCL. It is purely a
transport-layer feature that works on any Intel GPU (PVC, ARC, BMG/CRI/Xe3)
if the following conditions are met:

**Requirements:**
- `CCL_ATL_HMEM=1` (runtime, default off)
- `CCL_USE_HMEM=1` (runtime, default on, higher-level gate)
- Compiled with `ENABLE_OFI_HMEM=1` (default for dpcpp backend builds)
- `FI_PROVIDER` set to `verbs`, `cxi`, or `psm3`
- Provider must successfully negotiate `FI_HMEM` capability
- Linux kernel 5.12 or later with dmabuf support
- Intel GPU driver with dmabuf export support
- RDMA-capable NIC with verbs provider supporting `FI_HMEM_ZE`

BMG/CRI (device ID `0x6740`, family8) is not blocked from any of these paths.

### All GPU Direct Mechanisms in oneCCL

oneCCL provides five mechanisms for avoiding host staging. **None are enabled by
default.** All require explicit opt-in and have specific transport or stack requirements:

| Mechanism | Env Var | Transport | Default | BMG/CRI Compatible |
|---|---|---|---|---|
| **OFI HMEM** | `CCL_ATL_HMEM=1` | OFI (verbs/cxi/psm3) | Off | Yes |
| **Direct GPU RDMA** | `CCL_SYCL_ENABLE_DIRECT_GPU_RDMA=1` | MPI only | Off | Yes (not in blocklist) |
| **Pipeline GPU RDMA** | `CCL_SYCL_ENABLE_PIPELINE_GPU_RDMA=1` | MPI only | Off | Yes |
| **pt2pt offload** | `CCL_SEND=offload` / `CCL_RECV=offload` | OFI (PSM3_GPUDIRECT) or MPI | Off | Requires PSM3 |
| **MPI HMEM** | `CCL_ATL_HMEM=1` + `CCL_ATL_TRANSPORT=mpi` | MPI (sets I_MPI_OFFLOAD=2) | Off | Yes |

**OFI HMEM** is the primary mechanism for BMG/CRI with OFI transport (the recommended
transport for inference). The SYCL-based mechanisms (Direct GPU RDMA, Pipeline GPU RDMA)
are alternatives that work through MPI transport only and are gated by
`should_disable_rdma()`. BMG/CRI falls to the `default` case which does **not** disable RDMA.

Note: `CCL_SYCL_ENABLE_DIRECT_GPU_RDMA` is explicitly disabled when `CCL_ATL_TRANSPORT=ofi`
(the recommended transport). It only works via MPI transport with `I_MPI_OFFLOAD` set.

### GPU-to-GPU DMA Across Nodes (Scaleout with HMEM)

When HMEM is enabled on **both** nodes, the full scale-out data path achieves
GPU-to-GPU DMA without any host memory staging:

```
Without HMEM (default):
  Node A                                          Node B
  ┌───────┐   ┌────────┐   ┌───────┐        ┌───────┐   ┌────────┐   ┌───────┐
  │ GPU A │──▶│ Host A │──▶│ NIC A │──net──▶│ NIC B │──▶│ Host B │──▶│ GPU B │
  └───────┘D2H└────────┘   └───────┘        └───────┘   └────────┘H2D└───────┘
  4 PCIe traversals: GPU→Host (D2H) + Host→NIC + NIC→Host + Host→GPU (H2D)

With HMEM enabled on both sides:
  Node A                    Node B
  ┌───────┐   ┌───────┐        ┌───────┐   ┌───────┐
  │ GPU A │──▶│ NIC A │──net──▶│ NIC B │──▶│ GPU B │
  └───────┘   └───────┘        └───────┘   └───────┘
  2 PCIe traversals: GPU→NIC (NIC DMA-reads GPU) + NIC→GPU (NIC DMA-writes GPU)
  Host CPU still posts fi_tsendmsg/fi_trecvmsg but data never touches host DRAM.
```

Both the send path and receive path in `atl_ofi.cpp` use the HMEM MR cache for GPU
buffers:
- **Send** (line ~481): `fi_tsendmsg()` with MR obtained from `cache.get()`, the
  sending NIC DMA-reads directly from GPU A's memory
- **Recv** (line ~522): `fi_trecvmsg()` with MR obtained from `cache.get()`, the
  receiving NIC DMA-writes directly into GPU B's memory

This is functionally equivalent to NVIDIA GPUDirect RDMA but uses Linux dmabuf
instead of nvidia-peermem as the kernel interface.

**Requirements for scale-out GPU-to-GPU DMA:**

| Requirement                      | Why                                              |
|----------------------------------|--------------------------------------------------|
| `CCL_ATL_HMEM=1` on both nodes  | Enables HMEM MR registration on send and recv    |
| libfabric `FI_HMEM` on both     | Both provider instances must negotiate FI_HMEM   |
| GPU memory pre-registered       | `fi_mr_regattr` with FI_HMEM_ZE must succeed     |
| Linux dmabuf (kernel 5.12+)     | Backs FI_HMEM_ZE on both sender and receiver     |
| PCIe BAR large enough           | NIC needs BAR access to full GPU memory range    |
| NIC supports FI_HMEM_ZE         | Both NICs must handle device memory descriptors  |

**Comparison with NVIDIA GPUDirect RDMA:**

| Aspect               | NVIDIA GPUDirect RDMA       | Intel HMEM (PVC/BMG/CRI)         |
|----------------------|-----------------------------|------------------------------|
| Kernel interface     | nvidia-peermem module       | dmabuf (standard Linux)      |
| Provider API         | `FI_HMEM_CUDA`              | `FI_HMEM_ZE`                 |
| NIC reads GPU mem    | Yes (via P2P BAR)           | Yes (via P2P BAR + dmabuf)   |
| NIC writes GPU mem   | Yes (via P2P BAR)           | Yes (via P2P BAR + dmabuf)   |
| GPU initiates I/O    | Yes (GDRCopy, NVSHMEM)      | **No**, host CPU posts ops   |
| Maturity             | Production (10+ years)      | Experimental                 |

The last row is the remaining gap: on Intel Xe, the GPU kernel cannot autonomously
post RDMA operations. The host CPU must call `fi_tsendmsg`/`fi_trecvmsg`. But the
**data path** is GPU to NIC to network to NIC to GPU with zero host memory copies.

**Applies to both PVC and BMG/CRI.** Neither is in `should_disable_rdma()`. Only certain
ARC B-series desktop cards (0xE20B, 0xE20C, 0xE20D, 0xE212, 0xE220, 0xE221, 0xE223)
are blocked. PVC (family2, device mask 0xBD0) and BMG/CRI (family8, device 0x6740) both
fall to the `default` case which does **not** disable RDMA.

### Production Readiness

HMEM is **experimental** and not enabled by default because:
- Not all NIC and driver combinations support `FI_HMEM_ZE` reliably
- Memory registration overhead per collective (cached, but first-call penalty)
- Requires specific libfabric build (`--enable-verbs --with-ze`)
- Failure mode is silent hang (NIC cannot access GPU memory, waits indefinitely)

When it works, it eliminates one PCIe round-trip per inter-node message (2-5 us
saved per collective). For BMG/CRI's 160 allreduces per token, this could save 320-800 us
of TPOT if validated on the target stack.

---

## TMP_BUF: Async Collectives with Staging

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

```
Without TMP_BUF (default):

  User buf ────────────────────────────────────────────▶ Result in User buf
               ◀── buffer LOCKED ──▶
               (can't reuse until collective finishes)
  Time: ═══╪═══════════════════════════╪═══════════════▶
           start                     done


With TMP_BUF:

  User buf ──┐                                  ┌──▶ Result in User buf
         copy│                              copy│
             ▼                                  │
  TMP buf    ├────── collective runs ───────────┤
             │     (on internal buffer)         │
  User buf   └── FREE: reuse immediately ───────┘
                  (next layer writes here)
  Time: ═════╪══════════════════════════════════╪════▶
            start                             done
```

**Tradeoff:** adds two extra copies (in and out) but makes the collective fully
asynchronous from the caller's perspective. Useful when:
- You need to overlap collective with compute on the **same** buffer
- Latency tolerance is high but throughput matters (prefill, not decode)
- You cannot restructure code to use double-buffering

For decode (latency-critical), the extra copies usually hurt more than the
overlap helps. Double-buffering with `async_op=True` is preferred.

---

## Published Benchmarks

### Aurora Allreduce Latency (PVC, Xe Link, Slingshot-11)

Aurora is the DOE supercomputer at Argonne National Laboratory. It uses PVC GPU
tiles (predecessor to BMG/CRI), Xe Link for intra-node fabric, and HPE Slingshot-11
for the network. These numbers represent the best-case Intel GPU collective
performance with full hardware support.

From Ibeid et al., "Scaling MPI Applications on Aurora" (arXiv 2512.04291):

| Message Size | 1 Node (6 GPUs) | 2048 Nodes (~12K GPUs) |
|---|---|---|
| 8 B | ~15 us | ~250 us |
| 512 B | ~20 us | ~250 us |
| 2 KB | ~25 us | ~260 us |
| 64 KB | ~50 us | ~280 us |

Latency growth is sub-linear (recursive-doubling and tree algorithms). Aurora uses 8x
Slingshot-11 NICs per node (200 Gbps each), with 23-25 GB/s per NIC for GPU buffers.

### oneCCL vs Optimized Libraries (Pre-Production Aurora, 2024)

From Hidayetoglu et al., "HiCCL: A Hierarchical Collective Communication Library"
(arXiv 2408.05962), tested on 4 Aurora nodes (48 GPU tiles):

| Collective | oneCCL Throughput | HiCCL Throughput | Gap |
|---|---|---|---|
| Allreduce | 40-80 GB/s | 80-120 GB/s | 1.5-2x |
| Broadcast | 20-40 GB/s | 80-120 GB/s | 2-4x |
| All-to-All | 20-40 GB/s | 40-80 GB/s | 2x |

HiCCL reported a **12.1x geometric mean improvement** over oneCCL across all
collectives. Important caveats:
- This was pre-production Aurora with early oneCCL (2024)
- oneCCL's multi-NIC striping and hierarchical algorithms were not yet tuned for
  Aurora's 12-GPU, 8-NIC topology
- Production oneCCL has improved since then

### Large-Scale Training Efficiency

From Vooturi et al., "Scalable Pretraining of Large MoE Language Models on Aurora"
(arXiv 2604.00785):
- 220B parameter MoE model scaled from 384 to 12,288 PVC GPU tiles
- **90% scaling efficiency** at 12,288 tiles
- Communication overhead managed through expert-parallel sharding

### How These Numbers Apply to BMG/CRI

```
Aurora Node (PVC):                    BMG/CRI Node (Xe3):

  ┌──────────────────────────────┐    ┌─────────────────────────────────┐
  │  Tile0 ══Xe Link══ Tile1    │    │  GPU0 ──PCIe── CPU ──PCIe── GPU1│
  │    ║                  ║      │    │                 │                │
  │  Tile2 ══Xe Link══ Tile3    │    │  GPU2 ──PCIe── CPU ──PCIe── GPU3│
  │    ║                  ║      │    │               (UPI)              │
  │  Tile4 ══Xe Link══ Tile5    │    └────────────────┬────────────────┘
  │    ║                  ║      │                    │
  │  [8x Slingshot-11 NICs]      │                1-2 NICs
  └──────────────────────────────┘                    │
                                                   Network

  • 6 GPU tiles, Xe Link mesh         • 4 GPUs, PCIe only
  • 8 NICs x 200 Gbps = 1.6 Tbps     • 1-2 NICs x 200 Gbps
  • Scale-up: 100+ GB/s (Xe Link)     • Scale-up: 25-32 GB/s (PCIe)
  • Scaleout: host-staged but         • Scaleout: host-staged,
    8 NICs stripe bandwidth             fewer NICs, lower aggregate BW
```

BMG/CRI nodes are more constrained than Aurora:
- No Xe Link (Aurora has 28 GB/s per Xe Link between GPU stacks)
- Fewer NICs per node (Aurora has 8x Slingshot-11)
- PCIe-only intra-node path

Expect **higher per-collective latency** and **lower bandwidth** than the Aurora numbers
above. The Aurora MPI results are a useful PVC reference point, not a bound on optimized
collective implementations (HiCCL demonstrated 12× improvement over baseline oneCCL at
4 nodes). BMG/CRI without fabric will sit closer to the small-message latency floor
(15-25 us per collective) and will not scale bandwidth as well with message size.

---

## Practical Implications for BMG/CRI Deployment

TP=4 (tensor parallelism across 4 GPUs) is a common inference configuration: one
model layer is sharded across 4 GPUs, each doing a fraction of the GEMM, with an
allreduce at the end to combine results. A 2-layer transformer needs 2 allreduces
per transformer block (one after the attention GEMM, one after the MLP GEMM), so
80 layers need 160 allreduces per generated token.

```
Latency Breakdown: One Allreduce (TP=4, 2 nodes, 16 KB message):

  ┌────────┬──────────┬─────────┬─────────┬─────────┬─────────┬────────┐
  │        │          │         │         │         │         │        │
  │  D2H   │ Scale-up │  D2H    │  NIC    │ Network │  NIC    │  H2D   │
  │  copy  │ (PCIe    │  copy   │  post   │ transit │  recv   │  copy  │
  │ (GPU   │  P2P RS) │(staging)│  send   │         │         │ (GPU   │
  │ →Host) │          │         │         │         │         │ ←Host) │
  ├────────┼──────────┼─────────┼─────────┼─────────┼─────────┼────────┤
  │   2 us │    5 us  │   2 us  │   1 us  │   5 us  │   1 us  │   2 us │
  └────────┴──────────┴─────────┴─────────┴─────────┴─────────┴────────┘
  |◀─────────────────── Total: 25-40 us ────────────────────────────▶|

                    |◀── host staging tax ──────────────────────────▶|
                       11 us eliminated with HMEM or GPU Direct RDMA

  x 160 allreduces per token (80 layers x 2 allreduces each) = 1.8-3.5 ms TPOT budget
```

Given BMG/CRI's constraints (no fabric, no GPU-initiated network I/O):

1. **Never set `CCL_ALLREDUCE=ring` (or any non-topo value) for GPU inference.**
   This forces GPU→host→CPU-algorithm→host→GPU for every collective. Use
   `CCL_ALLREDUCE_SCALEOUT=ring` to control inter-node only.

2. **Verify P2P access works.** Without it, even intra-node communication falls
   back to host staging. Check `/dev/dri/` permissions and IOMMU settings.

3. **Pin ranks to match GPU NUMA affinity.** The `topo` algorithm's scale-up phase
   uses P2P DMA over PCIe. Misaligned pinning routes DMA through the wrong PCIe
   root complex, crossing UPI unnecessarily.

4. **Accept the host staging tax for scaleout.** In production (OFI transport,
   default config), every inter-node collective pays GPU-host-NIC-host-GPU.
   Budget 2-5 us per collective for the extra PCIe round-trips. `CCL_ATL_HMEM=1`
   can eliminate this in theory but is experimental and not production-ready.

5. **Use `async_op=True` to overlap.** While one collective is in the host staging
   phase, the GPU can run compute for the next layer — the main mechanism
   for hiding communication latency.

6. **Don't enable `CCL_ATL_HMEM=1` in production** unless explicitly validated on
   your driver/kernel/NIC stack. It's experimental and can cause silent hangs.

See [Perf Tuning](perf_tuning) for the full variable reference and launch templates.

---

## Future Hardware: UALink, GPU RDMA

| Constraint | BMG/CRI (Xe3) | Future (UALink-equipped) |
|---|---|---|
| GPU-initiated network I/O | No | Expected (HW RDMA engine) |
| NIC-to-GPU DMA (zero-copy) | Experimental (dmabuf) | Native |
| Intra-node fabric | None (PCIe only) | UALink (high-BW open mesh) |
| P2P atomics | Unreliable over PCIe | Full support expected |
| Copy engines for fabric | Main only (no link target) | Link engines + UALink |
| NIC integration | Discrete (PCIe-attached) | Closer integration expected |
| Host staging required? | **Yes** (default); experimental bypass via HMEM | No (with GPU-initiated RDMA) |

```
BMG/CRI (current):                          Future (UALink):

  ┌─────────────────────────┐          ┌───────────────────────────────────┐
  │ GPU0  GPU1  GPU2  GPU3  │          │  GPU0 ═══ GPU1 ═══ GPU2 ═══ GPU3  │
  │   │     │     │     │   │          │    ║        ║        ║        ║   │
  │   └─PCIe┴─PCIe┴─PCIe┘   │          │  UALink  UALink  UALink  UALink   │
  │          │              │          │    ║        ║        ║        ║   │
  │       CPU/UPI           │          │  GPU4 ═══ GPU5 ═══ GPU6 ═══ GPU7  │
  │          │              │          └─────────────────┬─────────────────┘
  │         NIC             │                            │
  └──────────┼──────────────┘                NIC (GPU-attached or CXL)
             │                                           │
          Network                                     Network

  • All comm goes through CPU            • GPU-to-GPU via UALink (~200+ GB/s)
  • Scaleout requires host staging       • GPU-initiated RDMA to NIC
  • PCIe BW ceiling: ~32 GB/s            • No host staging required
  • Copy engine orchestrated by host     • Full overlap: GPU posts RDMA + computes
```

When GPU-initiated network I/O and UALink arrive:
- Intra-node scale-up uses fabric instead of PCIe (100+ GB/s vs 32 GB/s)
- Scaleout collectives eliminate the host-NIC-host copy
- `topo` algorithm can pipeline GPU-initiated sends with compute
- Latency drops by 1-2 PCIe round-trips per collective (2-5 us each)
- `offload` mode for send/recv becomes viable (currently requires HW RDMA)

---

## References

### Hardware and Source
- [Intel Data Center GPU Max Series (PVC) Product Specs](https://www.intel.com/content/www/us/en/products/sku/232873/intel-data-center-gpu-max-1550/specifications.html) : Xe Link frequency, PCIe Gen5, tile architecture
- [oneCCL Source: `topo_manager.cpp`](https://github.com/oneapi-src/oneCCL/blob/master/src/topology/topo_manager.cpp) : `check_p2p_access()`, `check_p2p_atomics()`, fabric connectivity probing
- [oneCCL Source: `ze_primitives.cpp`](https://github.com/oneapi-src/oneCCL/blob/master/src/sched/entry/ze/ze_primitives.cpp) : `device_family` enum, `should_disable_rdma()`, copy engine selection
- [oneCCL Documentation: Environment Variables](https://github.com/oneapi-src/oneCCL/blob/master/doc/rst/source/env-variables.rst) : `CCL_ATL_HMEM`, `CCL_ZE_COPY_ENGINE`, `TMP_BUF`, `offload` mode
- [oneCCL Documentation: dmabuf Support](https://github.com/oneapi-src/oneCCL/blob/master/doc/rst/source/advanced-configuration/dmabuf.rst) : GPU memory registration with Linux dmabuf and OFI verbs
- [UALink Consortium](https://ualink.org/) : open standard for accelerator interconnects (Intel founding member)

### Benchmarks and Papers
- Ibeid et al., "Scaling MPI Applications on Aurora" (arXiv:2512.04291, Dec 2025) : MPI collective latency and bandwidth at 2048+ nodes on PVC
- Hidayetoglu et al., "HiCCL: A Hierarchical Collective Communication Library" (arXiv:2408.05962, Aug 2024) : oneCCL vs optimized collectives on pre-production Aurora
- Vooturi et al., "Scalable Pretraining of Large MoE Language Models on Aurora" (arXiv:2604.00785, Apr 2026) : 90% scaling efficiency at 12,288 PVC tiles
- Ma et al., "CoCoDiff: Optimizing Collective Communications for Distributed Diffusion Transformer Inference" (arXiv:2604.14561, Apr 2026) : 3.6x avg speedup for all-to-all on Aurora with topology-aware decomposition

---

## Appendix: BMG/CRI Device Recognition in oneCCL Source

This section is for readers debugging oneCCL behavior or checking whether a
specific GPU family gets a specific code path.

oneCCL recognizes BMG/CRI as device ID `0x6740` (device family `family8`). It is
classified as an "arc card" and uses SYCL kernel-based algorithms with
PCIe-oriented code paths. BMG/CRI is **not** in the `should_disable_rdma()`
blocklist — that function only blocks certain ARC B-series desktop cards
(0xE20B-0xE223) which had driver-level issues with GPU RDMA. All GPU RDMA
mechanisms (HMEM, Direct GPU RDMA, Pipeline GPU RDMA) are available for BMG/CRI.
The AOT compilation targets include `xe3` alongside `pvc` and `xe2`.

**Next:** [When to Use Which Collective](03_when_to_use) — the decision guide for
picking the right collective for each inference and training pattern. If you're here
for the hands-on work, start with [Environment Setup](../notebooks/02_environment_setup)
to verify your stack before running any benchmark.
