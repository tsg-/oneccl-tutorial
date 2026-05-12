# GPU Hardware Constraints

This chapter documents the hardware-level constraints that shape how oneCCL
implements collectives on CRI (Crescent Island, Xe3) — and provides historical
context from PVC (Ponte Vecchio, Xe-HPC) which introduced similar challenges.

---

## The Two Constraints That Matter

| Constraint | PVC (Xe-HPC, 2022) | CRI (Xe3, 2026) |
|---|---|---|
| Intra-node GPU fabric | Xe Link (high-BW coherent) | **None — PCIe only** |
| GPU-initiated network I/O | No | No |
| NIC-to-GPU DMA (HMEM/dmabuf) | Experimental (`CCL_ATL_HMEM=1`) | Experimental (`CCL_ATL_HMEM=1`) |
| `offload` mode (Xe Link RDMA) | Yes (intra-node only) | No (no fabric) |
| **Scaleout default** | **Host-staged** | **Host-staged** |

**CRI is more constrained than PVC** in topology: PVC had Xe Link for fast intra-node
GPU-to-GPU transfers (used on systems like Aurora with 6 GPUs per node) and could use
`offload` mode for send/recv over that fabric. CRI has no GPU fabric — every GPU-to-GPU
transfer, even within the same node, traverses PCIe and potentially UPI.

Neither generation supports **GPU-initiated network I/O** — the GPU cannot autonomously
post sends or receives to the NIC. The host CPU always orchestrates the transfer.

Both generations have an **experimental** path for NIC-to-GPU DMA via `CCL_ATL_HMEM=1`
(the NIC reads/writes GPU memory directly via dmabuf). However, this is **off by
default and not production-validated**. In standard deployments, all inter-node
(scaleout) traffic is host-staged — this is the primary scalability limitation of
Intel Xe GPUs compared to NVIDIA's production-grade GPUDirect RDMA.

### CRI in oneCCL Source

oneCCL recognizes CRI as device ID `0x6740` (device family `family8`). It is classified
as an "arc card" and uses SYCL kernel-based algorithms with PCIe-oriented code paths.
CRI is **not** in the `should_disable_rdma()` blocklist — all GPU RDMA mechanisms
(HMEM, Direct GPU RDMA, Pipeline GPU RDMA) are available. The AOT compilation targets
include `xe3` alongside `pvc` and `xe2`.

---

## GPU-Initiated Network I/O: Absent on CRI

The term "GPU Direct" covers a spectrum of capabilities. The critical distinction is
who **initiates** the network transfer:

| Capability | Description | NVIDIA | PVC (Xe-HPC) | CRI (Xe3) |
|---|---|---|---|---|
| **GPU-initiated network I/O** | GPU kernel autonomously posts RDMA sends/receives to NIC | Yes (GPUDirect Async) | No | No |
| **NIC-to-GPU DMA (host-orchestrated)** | NIC reads/writes GPU memory via PCIe; host CPU sets up transfers | Yes (GPUDirect RDMA) | Experimental (HMEM) | Experimental (HMEM) |
| **GPU-to-GPU P2P DMA** | One GPU's copy engine reads another GPU's memory | Yes (NVLink / PCIe) | Yes (Xe Link + PCIe) | PCIe only |
| **GPU RDMA offload** | Library offloads send/recv to device-side agent | Yes | Xe Link only (`offload` mode) | No (no fabric) |

On NVIDIA hardware, a GPU kernel can **autonomously** initiate network operations — no
host CPU involvement after setup. On Intel Xe (both PVC and CRI), the host CPU must
orchestrate every network transfer. The GPU can perform local DMA (copy engines for P2P),
but cannot issue commands to the NIC.

oneCCL has an **experimental** path for NIC-to-GPU DMA on both PVC and CRI via the HMEM
mechanism (`CCL_ATL_HMEM=1`). When enabled, the NIC reads/writes GPU memory directly
(via libfabric `FI_HMEM_ZE` backed by Linux dmabuf), eliminating the host staging buffer
on the data path. The host CPU still initiates the transfer, but data does not transit
through host memory. This is **off by default** — see
[NIC-to-GPU DMA via HMEM](#nic-to-gpu-dma-via-hmem-ccl_atl_hmem--experimental) for details.

PVC with Xe Link additionally supports `offload` mode for send/recv, where the library
uses Xe Link fabric for intra-node transfers without host staging. CRI lacks this — no
fabric means no `offload` path for intra-node communication.

On CRI, the GPU has no path to initiate network I/O:

```
  ┌──────────────────────┐
  │    CRI GPU (Xe3)     │
  ├──────────────────────┤
  │  Compute (EUs)       │
  │  Copy Engines        │──── Level Zero IPC (intra-node P2P via PCIe)
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

This means:

1. **The host CPU is always on the critical path** — it must post every send/recv
   to the NIC (no GPU-initiated network I/O)
2. **In production (default config):** every scaleout collective requires a
   GPU→host copy before network transfer, and host→GPU copy after arrival
3. **With HMEM (experimental):** NIC reads/writes GPU memory directly, skipping
   host DRAM — but this path is not validated for production use

This host-staging requirement is the **primary scaleout bottleneck** on both PVC
and CRI. It adds 2-5 µs per collective and limits effective cross-node bandwidth
to what the PCIe link between GPU and host can sustain.

---

## CRI vs PVC: Absence of Intra-Node Fabric

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

## The `topo` Algorithm: Host-Staged Hierarchical Communication

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
  │ GPU0  GPU1  GPU2  GPU3    │       │ GPU4  GPU5  GPU6  GPU7    │
  └───────────────────────────┘       └───────────────────────────┘
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
                 compute and DMA overlap     no stalls — GPU always busy


Timeline (sync, no overlap — DEFAULT without async_op):

              Layer N       wait   Layer N+1     wait   Layer N+2
              ───────────── ────── ───────────── ────── ─────────
Compute EUs:  ████ GEMM N █        ████ GEMM N+1        ████ GEMM █
Copy Engine:               ░░ AR N ░░           ░░AR N+1░░
              ├─────────────┤├─────┤├────────────┤├─────┤
                            ▲ STALL ▲            ▲ STALL▲
                   GPU idle while allreduce completes
```

The async path hides communication latency behind compute. For decode (small GEMMs,
~50 µs), the overlap window is tight — communication must complete within one layer's
compute time or it stalls the next layer.

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

The result is a P2P connectivity matrix like:

```
P2P Connectivity Matrix (CRI 4-GPU node, 2 sockets):

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

## Scaleout Limitation: Host Staging Is the Default

Both PVC and CRI **always host-stage inter-node traffic by default**. This is the
scalability limitation you will encounter in production. oneCCL has two algorithm
families, and both default to host staging:

**1. The `topo` algorithm (default for GPU buffers):**

In `coll_util.cpp`, the scaleout phase explicitly copies to host when HMEM is
not enabled (the default):
```
if (!enable_hmem) {
    LOG_DEBUG("topo/scale_out: use host_...");
    // D2H copy into staging buffer
}
```

**2. SYCL scaleout kernels (called from topo's scale-up path):**

In `allreduce_scaleout_sycl.cpp`, GPU RDMA is explicitly disabled for OFI transport:
```cpp
bool copy_to_host = sycl_enable_direct_gpu_rdma ? false : true;
if (should_disable_rdma(ze_dev) || atl_transport == ccl_atl_ofi) {
    copy_to_host = true;  // OFI always host-stages
}
```

The comment in `sycl_coll_base.cpp` at `check_mpi_supports_rdma()` states:
`"ofi collective only supports host memory"`.

**Why OFI can't do GPU RDMA in the SYCL kernel path:** The SYCL scaleout kernels
use direct `atl_comm->allreduce()` calls from a host task. The OFI ATL layer's
collective implementation does not use HMEM MR registration for these internal
collective calls — it only handles point-to-point sends/recvs with HMEM. The
collective-level ATL calls always expect host-accessible buffers.

**Result:** `CCL_SYCL_ENABLE_DIRECT_GPU_RDMA=1` only works with **MPI transport**
(requires `I_MPI_OFFLOAD=2` with Intel MPI, or MPICH with
`MPIR_CVAR_CH4_OFI_ENABLE_HMEM=1`). OFI transport — the recommended transport
for inference — cannot use this path.

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

## NIC-to-GPU DMA via HMEM (CCL_ATL_HMEM) — Experimental

The **only mechanism** for avoiding host staging with OFI transport is
`CCL_ATL_HMEM=1`. When set, oneCCL registers GPU memory directly with the
libfabric transport layer using the `FI_HMEM` capability. This allows the NIC
to perform RDMA reads/writes directly from/to GPU memory, eliminating the
host staging copy on the send and receive paths.

```
Without HMEM (current default):
  Send: GPU buf → copy to host staging → NIC reads host → network
  Recv: network → NIC writes host → copy to GPU buf

With HMEM enabled:
  Send: GPU buf → NIC reads GPU memory directly → network
  Recv: network → NIC writes GPU memory directly
  (host CPU still orchestrates, but data path skips host memory)
```

### How It Works Internally

The mechanism in oneCCL (from `src/atl/ofi/atl_ofi.cpp`):

1. At init, oneCCL opens a libfabric provider with `FI_HMEM` capability and
   `FI_MR_HMEM` memory registration mode
2. On each send/recv, oneCCL calls `zeMemGetAllocProperties()` to determine if
   the buffer is GPU memory
3. For GPU buffers, it registers with `fi_mr_regattr()` using `iface=FI_HMEM_ZE`
   and the Level Zero device index
4. The registered MR descriptor is passed to `fi_tsendmsg()`/`fi_trecvmsg()`
5. The libfabric verbs provider internally uses Linux dmabuf (kernel ≥ 5.12) to
   allow the NIC to DMA from/to the GPU BAR

```
HMEM Registration and Transfer Flow:

  ┌───────────────────────────────────────────────────────────────┐
  │                      oneCCL (atl_ofi.cpp)                     │
  │                                                               │
  │  1. zeMemGetAllocProperties(buf)                              │
  │     → "this is ZE device memory on device idx N"              │
  │                                                               │
  │  2. fi_mr_regattr(buf, len, iface=FI_HMEM_ZE, device=N)      │
  │     → MR handle (cached for subsequent calls)                 │
  │                                                               │
  │  3. fi_tsendmsg(ep, msg, MR) / fi_trecvmsg(ep, msg, MR)      │
  │     → NIC uses MR to locate GPU BAR mapping                   │
  └─────────────────────────────┬─────────────────────────────────┘
                                │
  ┌─────────────────────────────▼─────────────────────────────────┐
  │                  libfabric (verbs provider)                    │
  │                                                               │
  │  fi_mr_regattr → ibv_reg_dmabuf_mr(dmabuf_fd, offset, len)   │
  │  fi_tsendmsg   → ibv_post_send(wr with GPU-mapped lkey)      │
  │  fi_trecvmsg   → ibv_post_recv(wr with GPU-mapped lkey)      │
  └─────────────────────────────┬─────────────────────────────────┘
                                │
  ┌─────────────────────────────▼─────────────────────────────────┐
  │                  NIC Hardware (RDMA engine)                    │
  │                                                               │
  │  DMA read from GPU BAR ──→ wire ──→ DMA write to GPU BAR     │
  │  (no host memory involved in data path)                       │
  └───────────────────────────────────────────────────────────────┘
```

oneCCL never passes dmabuf file descriptors directly — the dmabuf mechanism is
abstracted behind libfabric's `FI_HMEM` API.

### No Hardware-Specific Gating

HMEM has **no device-family restrictions** in oneCCL. It is purely a
transport-layer feature — it works on any Intel GPU (PVC, ARC, CRI/Xe3)
if the following conditions are met:

**Requirements:**
- `CCL_ATL_HMEM=1` (runtime, default off)
- `CCL_USE_HMEM=1` (runtime, default on — higher-level gate)
- Compiled with `ENABLE_OFI_HMEM=1` (default for dpcpp backend builds)
- `FI_PROVIDER` set to `verbs`, `cxi`, or `psm3`
- Provider must successfully negotiate `FI_HMEM` capability
- Linux kernel ≥ 5.12 with dmabuf support
- Intel GPU driver with dmabuf export support
- RDMA-capable NIC with verbs provider supporting `FI_HMEM_ZE`

CRI (device ID `0x6740`, family8) is **not blocked** from any of these paths.

### All GPU Direct Mechanisms in oneCCL

oneCCL provides five mechanisms for avoiding host staging. **None are enabled by
default.** All require explicit opt-in and have specific transport/stack requirements:

| Mechanism | Env Var | Transport | Default | CRI Compatible |
|---|---|---|---|---|
| **OFI HMEM** | `CCL_ATL_HMEM=1` | OFI (verbs/cxi/psm3) | Off | Yes |
| **Direct GPU RDMA** | `CCL_SYCL_ENABLE_DIRECT_GPU_RDMA=1` | MPI only | Off | Yes (not in blocklist) |
| **Pipeline GPU RDMA** | `CCL_SYCL_ENABLE_PIPELINE_GPU_RDMA=1` | MPI only | Off | Yes |
| **pt2pt offload** | `CCL_SEND=offload` / `CCL_RECV=offload` | OFI (PSM3_GPUDIRECT) or MPI | Off | Requires PSM3 |
| **MPI HMEM** | `CCL_ATL_HMEM=1` + `CCL_ATL_TRANSPORT=mpi` | MPI (sets I_MPI_OFFLOAD=2) | Off | Yes |

**OFI HMEM** is the primary mechanism for CRI with OFI transport (the recommended
transport for inference). The SYCL-based mechanisms (Direct GPU RDMA, Pipeline GPU RDMA)
are alternatives that work through MPI transport only and are gated by
`should_disable_rdma()` — CRI falls to the `default` case which does **not** disable RDMA.

Note: `CCL_SYCL_ENABLE_DIRECT_GPU_RDMA` is explicitly disabled when `CCL_ATL_TRANSPORT=ofi`
(the recommended transport). It only works via MPI transport with `I_MPI_OFFLOAD` set.

### GPU-to-GPU DMA Across Nodes (Scale-Out with HMEM)

When HMEM is enabled on **both** nodes, the full scale-out data path achieves
GPU-to-GPU DMA without any host memory staging:

```
Without HMEM (default):
  Node A                                          Node B
  ┌──────┐   ┌──────┐   ┌─────┐       ┌─────┐   ┌──────┐   ┌──────┐
  │GPU A │──▶│Host A│──▶│NIC A│──net──▶│NIC B│──▶│Host B│──▶│GPU B │
  └──────┘D2H└──────┘   └─────┘       └─────┘   └──────┘H2D└──────┘
  4 PCIe traversals: GPU→Host (D2H) + Host→NIC + NIC→Host + Host→GPU (H2D)

With HMEM enabled on both sides:
  Node A                    Node B
  ┌──────┐   ┌─────┐       ┌─────┐   ┌──────┐
  │GPU A │──▶│NIC A│──net──▶│NIC B│──▶│GPU B │
  └──────┘   └─────┘       └─────┘   └──────┘
  2 PCIe traversals: GPU→NIC (NIC DMA-reads GPU) + NIC→GPU (NIC DMA-writes GPU)
  Host CPU still posts fi_tsendmsg/fi_trecvmsg but data never touches host DRAM.
```

Both the send path and receive path in `atl_ofi.cpp` use the HMEM MR cache for GPU
buffers. Specifically:
- **Send** (line ~481): `fi_tsendmsg()` with MR obtained from `cache.get()` — the
  sending NIC DMA-reads directly from GPU A's memory
- **Recv** (line ~522): `fi_trecvmsg()` with MR obtained from `cache.get()` — the
  receiving NIC DMA-writes directly into GPU B's memory

This is functionally equivalent to NVIDIA GPUDirect RDMA, but uses Linux dmabuf
instead of nvidia-peermem as the kernel interface.

**Requirements for scale-out GPU-to-GPU DMA:**

| Requirement                      | Why                                              |
|----------------------------------|--------------------------------------------------|
| `CCL_ATL_HMEM=1` on both nodes  | Enables HMEM MR registration on send AND recv    |
| libfabric `FI_HMEM` on both     | Both provider instances must negotiate FI_HMEM   |
| GPU memory pre-registered       | `fi_mr_regattr` with FI_HMEM_ZE must succeed     |
| Linux dmabuf (kernel >= 5.12)   | Backs FI_HMEM_ZE on both sender and receiver     |
| PCIe BAR large enough           | NIC needs BAR access to full GPU memory range    |
| NIC supports FI_HMEM_ZE         | Both NICs must handle device memory descriptors  |

**Comparison with NVIDIA GPUDirect RDMA:**

| Aspect               | NVIDIA GPUDirect RDMA       | Intel HMEM (PVC/CRI)         |
|----------------------|-----------------------------|------------------------------|
| Kernel interface     | nvidia-peermem module       | dmabuf (standard Linux)      |
| Provider API         | `FI_HMEM_CUDA`              | `FI_HMEM_ZE`                 |
| NIC reads GPU mem    | Yes (via P2P BAR)           | Yes (via P2P BAR + dmabuf)   |
| NIC writes GPU mem   | Yes (via P2P BAR)           | Yes (via P2P BAR + dmabuf)   |
| GPU initiates I/O    | Yes (GDRCopy, NVSHMEM)      | **No** — host CPU posts ops  |
| Maturity             | Production (10+ years)      | Experimental                 |

The last row is the remaining gap: on Intel Xe, the GPU kernel cannot autonomously
post RDMA operations. The host CPU must call `fi_tsendmsg`/`fi_trecvmsg`. But the
**data path** is GPU → NIC → network → NIC → GPU with zero host memory copies.

**Applies to both PVC and CRI.** Neither is in `should_disable_rdma()` — only certain
ARC B-series desktop cards (0xE20B, 0xE20C, 0xE20D, 0xE212, 0xE220, 0xE221, 0xE223)
are blocked. PVC (family2, device mask 0xBD0) and CRI (family8, device 0x6740) both
fall to the `default` case which does **not** disable RDMA.

### Production Readiness

HMEM is **experimental** and not enabled by default because:
- Not all NIC/driver combinations support `FI_HMEM_ZE` reliably
- Memory registration overhead per collective (cached, but first-call penalty)
- Requires specific libfabric build (`--enable-verbs --with-ze`)
- Failure mode is silent hang (NIC cannot access GPU memory → indefinite wait)

When it works, it eliminates one PCIe round-trip per inter-node message (~2-5 µs
saved per collective). For CRI's 160 allreduces per token, this could save 320-800 µs
of TPOT — significant if validated on the target stack.

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

**Tradeoff:** Adds two extra copies (in + out) but makes the collective fully
asynchronous from the user's perspective. Useful when:
- You need to overlap collective with compute on the **same** buffer
- Latency tolerance is high but throughput matters (prefill, not decode)
- You cannot restructure code to use double-buffering

For decode (latency-critical), the extra copies usually hurt more than the
overlap helps. Double-buffering with `async_op=True` is preferred.

---

## Future Hardware: UALink and GPU RDMA

| Constraint | CRI (Xe3) | Future (UALink-equipped) |
|---|---|---|
| GPU-initiated network I/O | No | Expected (HW RDMA engine) |
| NIC-to-GPU DMA (zero-copy) | Experimental (dmabuf) | Native |
| Intra-node fabric | None (PCIe only) | UALink (high-BW open mesh) |
| P2P atomics | Unreliable over PCIe | Full support expected |
| Copy engines for fabric | Main only (no link target) | Link engines + UALink |
| NIC integration | Discrete (PCIe-attached) | Closer integration expected |
| Host staging required? | **Yes** (default); experimental bypass via HMEM | No (with GPU-initiated RDMA) |

```
CRI (current):                          Future (UALink):

  ┌─────────────────────────┐          ┌───────────────────────────────────┐
  │ GPU0  GPU1  GPU2  GPU3  │          │ GPU0 ═══ GPU1 ═══ GPU2 ═══ GPU3  │
  │   │     │     │     │   │          │   ║        ║        ║        ║   │
  │   └─PCIe┴─PCIe┴─PCIe┘  │          │ UALink  UALink  UALink  UALink   │
  │          │              │          │   ║        ║        ║        ║   │
  │       CPU/UPI           │          │ GPU4 ═══ GPU5 ═══ GPU6 ═══ GPU7  │
  │          │              │          └─────────────────┬─────────────────┘
  │         NIC             │                           │
  └──────────┼──────────────┘               NIC (GPU-attached or CXL)
             │                                          │
          Network                                    Network

  • All comm goes through CPU            • GPU-to-GPU via UALink (~200+ GB/s)
  • Scaleout requires host staging       • GPU-initiated RDMA to NIC
  • PCIe BW ceiling: ~32 GB/s            • No host staging required
  • Copy engine orchestrated by host     • Full overlap: GPU posts RDMA + computes
```

When GPU-initiated network I/O and UALink arrive:
- Intra-node scale-up uses fabric instead of PCIe — bandwidth jumps dramatically
- Scaleout collectives eliminate the host→NIC→host copy
- `topo` algorithm can pipeline GPU-initiated sends with compute
- Latency drops by 1-2 PCIe round-trips per collective (~2-5 µs each)
- `offload` mode for send/recv becomes viable (currently requires HW RDMA)

---

## Practical Implications for CRI Deployment

```
Latency Breakdown: One Allreduce (TP=4, 2 nodes, 16 KB message):

  ┌───────┬─────────┬─────────┬─────────┬─────────┬─────────┬───────┐
  │       │         │         │         │         │         │       │
  │ D2H   │Scale-up │  D2H    │  NIC    │ Network │  NIC    │  H2D  │
  │ copy  │(PCIe    │  copy   │  post   │ transit │  recv   │  copy │
  │       │ P2P RS) │(staging)│  send   │         │         │       │
  │       │         │         │         │         │         │       │
  ├───────┼─────────┼─────────┼─────────┼─────────┼─────────┼───────┤
  │ ~2 us │  ~5 us  │  ~2 us  │  ~1 us  │  ~5 us  │  ~1 us  │ ~2 us │
  └───────┴─────────┴─────────┴─────────┴─────────┴─────────┴───────┘
  |◀─────────────────── Total: ~25-40 us ───────────────────────────▶|

                    |◀── host staging tax ──────────────────────────▶|
                       ~11 us eliminated with HMEM/GPU Direct RDMA

  × 160 allreduces per token (80 layers × 2 row-parallel) = 1.8-3.5 ms TPOT budget
```

Given CRI's constraints (no fabric, no GPU-initiated network I/O):

1. **Never set `CCL_ALLREDUCE=ring` (or any non-topo value) for GPU inference.**
   This forces GPU→host→CPU-algorithm→host→GPU for every collective. Use
   `CCL_ALLREDUCE_SCALEOUT=ring` to control inter-node only.

2. **Verify P2P access works.** Without it, even intra-node communication falls
   back to host staging. Check `/dev/dri/` permissions and IOMMU settings.

3. **Pin ranks to match GPU NUMA affinity.** The `topo` algorithm's scale-up phase
   uses P2P DMA over PCIe — misaligned pinning routes DMA through the wrong PCIe
   root complex, crossing UPI unnecessarily.

4. **Accept the host staging tax for scaleout.** In production (OFI transport,
   default config), every inter-node collective pays GPU→host→NIC→host→GPU.
   Budget ~2-5 µs per collective for the extra PCIe round-trips. `CCL_ATL_HMEM=1`
   can theoretically eliminate this, but is experimental and not production-ready.

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

```
Aurora Node (PVC):                    CRI Node (Xe3):

  ┌──────────────────────────────┐    ┌─────────────────────────────────┐
  │  Tile0 ══Xe Link══ Tile1    │    │  GPU0 ──PCIe── CPU ──PCIe── GPU1│
  │    ║                  ║      │    │                 │                │
  │  Tile2 ══Xe Link══ Tile3    │    │  GPU2 ──PCIe── CPU ──PCIe── GPU3│
  │    ║                  ║      │    │               (UPI)              │
  │  Tile4 ══Xe Link══ Tile5    │    └────────────────┬────────────────┘
  │    ║                  ║      │                    │
  │  [8× Slingshot-11 NICs]     │                1-2 NICs
  └──────────────────────────────┘                    │
                                                   Network

  • 6 GPU tiles, Xe Link mesh         • 4 GPUs, PCIe only
  • 8 NICs × 200 Gbps = 1.6 Tbps     • 1-2 NICs × 200 Gbps
  • Scale-up: ~100+ GB/s (Xe Link)    • Scale-up: ~25-32 GB/s (PCIe)
  • Scaleout: host-staged but         • Scaleout: host-staged,
    8 NICs stripe bandwidth             fewer NICs, lower aggregate BW
```

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
