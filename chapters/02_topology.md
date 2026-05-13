# Topology and Algorithm Selection

## CRI Hardware Reality: NUMA-Only

CRI inference nodes use **NUMA-only intra-node topology** -- no UALink, no XeLink fabric.
This is the single most important constraint driving algorithm selection.

```
Socket 0 (NUMA 0)          Socket 1 (NUMA 1)
┌──────────────────┐        ┌──────────────────┐
│  GPU0  GPU1      │        │  GPU2  GPU3      │
│   │     │        │        │   │     │        │
│  PCIe Root Cx    │──UPI───│  PCIe Root Cx    │
│        │         │        │        │         │
│      CPU0        │        │      CPU1        │
└──────────────────┘        └──────────────────┘
```

Every GPU-to-GPU transfer crosses PCIe and potentially UPI. There is no direct peer path.

### PCIe Gen5 Bandwidth Budget

Each PCIe Gen5 x16 link provides:

| Direction | Theoretical | Achievable (with protocol overhead) |
|---|---|---|
| Unidirectional | 63 GB/s | ~50-55 GB/s |
| Bidirectional (full duplex) | 126 GB/s | ~90-100 GB/s |

For a P2P DMA between two GPUs on the same socket:
```
GPU0 → PCIe Root Complex → GPU1
Effective BW: ~25-32 GB/s (one direction, sharing root complex)
```

For a P2P DMA between GPUs on different sockets:
```
GPU0 → PCIe RC (Socket 0) → UPI → PCIe RC (Socket 1) → GPU2
Effective BW: limited by UPI (~50-100 GB/s shared across all traffic)
Added latency: ~200-400 ns per UPI hop
```

The UPI link is shared by all cross-socket traffic — CPU cache coherence, memory
accesses, and PCIe forwarding all compete for the same bandwidth. Under load from
multiple GPU pairs communicating simultaneously, effective per-pair bandwidth can
drop to 10-15 GB/s.

### Why P2P DMA Works (or Fails) on CRI

Level Zero IPC handles enable P2P DMA over PCIe, but several hardware/software
conditions must be met:

| Requirement | What Happens if Not Met |
|---|---|
| IOMMU in passthrough or disabled | DMA remapping adds ~1-2 µs per transfer |
| ACS (Access Control Services) disabled on PCIe bridges | P2P routed through CPU instead of direct switch |
| BAR (Base Address Register) large enough | GPU memory not fully mappable by peers |
| `/dev/dri/renderD*` permissions | Level Zero cannot open IPC handles |
| Same PCIe root complex (ideal) | Cross-root transfers add latency |

oneCCL's `topo_manager` probes these conditions at `init_process_group()` time and
builds a P2P connectivity matrix. If any pair fails the check, that pair falls back
to host-staged copy (GPU → host buffer → GPU), adding ~2-5 µs and halving bandwidth.

Verify P2P connectivity:
```bash
# Check if Level Zero reports peer access
ze_peak --peer-access

# Check IOMMU mode
dmesg | grep -i "iommu\|DMAR"
# Should see: "DMAR: IOMMU disabled" or "intel_iommu=off"

# Check ACS on PCIe bridges
setpci -s <bridge_bdf> ECAP_ACS+6.w
# Bit 2 (ACS P2P Request Redirect) should be 0 for direct P2P
```

---

## One-Shot Allreduce Under NUMA

One-Shot Allreduce requires simultaneous fan-out from each GPU to all N-1 peers.
Under NUMA:

- No direct GPU-GPU link -- every message bounces through CPU memory
- Simultaneous N-way fan-out saturates PCIe upstream bandwidth
- Cross-socket messages compete for UPI bandwidth (~64-100 GB/s shared)

**At N=8, One-Shot under NUMA is not 1-step -- it serializes on PCIe.**
Ring Allreduce with topology-aware construction is the correct algorithm here.

## Topology-Aware Ring Construction

The examples below use **two different node configurations** to illustrate the principle
clearly. The opening diagram of this chapter used 4 GPUs (2 per socket, matching the
CRI P2P matrix). The ring example below uses 8 GPUs (4 per socket) because the
interleaving failure mode is more visible with more ranks; the principle is identical
for 4 GPUs.

**8-GPU node (4 GPUs per socket):** The optimal ring minimizes UPI crossings to exactly 2:

```
Ring order:  GPU0 → GPU1 → GPU2 → GPU3 → GPU4 → GPU5 → GPU6 → GPU7
             └──── socket0 ────┘  ↑UPI↑  └──── socket1 ────┘
                                  └─────────────────────────────┘ (wrap)
```

This means 6 of 8 hops stay on-socket (fast PCIe), only 2 cross UPI.
A naively ordered ring (0,4,1,5,2,6,3,7) would alternate sockets on every hop -- 8x the UPI traffic.

The difference in hop cost (abbreviated to 2 GPUs per socket for diagram clarity):

```
Topology-aware ring (socket-local hops dominate):

  Socket 0                    Socket 1
  ┌────────────────────┐      ┌────────────────────┐
  │ GPU0 ──→ GPU1      │      │ GPU2 ──→ GPU3      │
  │  (PCIe: ~32 GB/s)  │      │  (PCIe: ~32 GB/s)  │
  └─────────┬──────────┘      └──────────┬─────────┘
            │                             │
            └──────── UPI (~50 GB/s) ─────┘
                    (only 2 hops cross)

Naive interleaved ring (every hop crosses UPI):

  GPU0 ──UPI──→ GPU4 ──UPI──→ GPU1 ──UPI──→ GPU5 ──UPI──→ ...
          ×8 UPI crossings = saturates inter-socket link
```

**oneCCL constructs topology-aware rings automatically when NUMA affinity is set correctly.**
The key is ensuring MPI rank-to-socket pinning matches GPU assignment:

```bash
# 8-GPU: ranks 0-3 on socket 0, ranks 4-7 on socket 1
I_MPI_PIN_DOMAIN=socket mpirun -n 8 -ppn 8 python train.py
# 4-GPU (2 per socket): same flag, ranks 0-1 on socket 0, ranks 2-3 on socket 1
I_MPI_PIN_DOMAIN=socket mpirun -n 4 -ppn 4 python train.py

# Verify ring construction
CCL_LOG_LEVEL=info python script.py 2>&1 | grep -i "ring order"
```

## Algorithm Decision Tree for CRI Inference

```
What collective do you need?
│
├── Allreduce (TP layer sync)
│     ├── msg < 1MB, TP <= 8  →  Ring              ✓ Done
│     ├── msg > 1MB           →  Ring (pipelined)  ✓ Done
│     └── One-Shot            →  P1 (requires GPU fabric / UALink, not available on CRI)
│
├── Allgather (sequence parallelism)
│     └── any size            →  Ring              ✓ Done
│
├── Alltoall (MoE expert routing)
│     ├── scale-up (intra-node) →  topo           ✓ Done
│     └── scale-out (inter-node) → scatter        ✓ Done
│                                  (Direct/Ring = X, gap)
│
├── Scatter/Gather (KV routing)
│     ├── intra-node          →  Ring P0 (coming)
│     └── inter-node / large  →  Use NIXL instead
│
└── Broadcast (control plane)
      └── small msgs          →  Ring             ✓ Done
```

## Message Size Regimes on NUMA

Unlike NVLink hardware where One-Shot wins below ~1MB, NUMA flips the crossover:

| Message Size | NUMA-Optimal Algorithm | Reason |
|---|---|---|
| < 64 KB | Ring (small-message path) | Low latency, fits in L3/IOLLC |
| 64 KB – 4 MB | Ring (pipelined RS+AG) | Pipelined reduce-scatter + allgather fills PCIe bandwidth |
| > 4 MB | Ring (large-message path) | Bandwidth-bound, ring optimal |
| Any | NOT One-Shot | Fan-out saturates PCIe |

### Ring Pipeline Efficiency on NUMA

For the ring algorithm on p ranks with message size n bytes:

```
Total data moved per rank = 2 × ((p-1)/p) × n
  (reduce-scatter: (p-1)/p × n sent + received)
  (allgather:      (p-1)/p × n sent + received)

Pipeline utilization (fraction of peak link BW achieved):
  Ideal: ((p-1)/p)  → approaches 1.0 as p grows
  p=4: 75%
  p=8: 87.5%

Time for ring allreduce (bandwidth-bound regime):
  T = 2(p-1)α + 2((p-1)/p) × n/BW_link

  For p=4, n=1MB, BW_link=25 GB/s (PCIe P2P achievable):
  T_bw = 2 × (3/4) × 1MB / 25 GB/s = 60 µs (bandwidth component)
  T_lat = 2 × 3 × α ≈ 6 × 5µs = 30 µs (latency component, α≈5µs for PCIe P2P)
  T_total ≈ 90 µs

  Same with UPI crossing (BW_link drops to ~15 GB/s effective):
  T_bw = 2 × (3/4) × 1MB / 15 GB/s = 100 µs
  T_total ≈ 130 µs  (44% regression from NUMA misalignment)
```

This quantifies why NUMA pinning is the single biggest lever: misalignment doesn't just
add latency to 2 hops — it degrades the effective bandwidth of every hop that crosses
UPI, compounding across all (p-1) ring steps.

## Practical Verification

Check your NUMA topology before deploying:

```bash
# GPU NUMA affinity
for i in 0 1 2 3; do
    numa=$(cat /sys/bus/pci/devices/$(ls -la /dev/dri/renderD$((128+i)) \
           | awk '{print $NF}' | xargs dirname | xargs dirname \
           | xargs basename)/numa_node 2>/dev/null || echo "unknown")
    echo "GPU$i -> NUMA node $numa"
done

# UPI bandwidth (theoretical ceiling for cross-socket traffic)
numactl --hardware | grep -E "node|distance"
```

Expected healthy output for a 2S server:
```
GPU0 -> NUMA node 0
GPU1 -> NUMA node 0
GPU2 -> NUMA node 1
GPU3 -> NUMA node 1

node distances:
node   0   1
  0:  10  21
  1:  21  10
```
