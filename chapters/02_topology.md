# Topology & Algorithm Selection

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

## One-Shot Allreduce Under NUMA

One-Shot Allreduce requires simultaneous fan-out from each GPU to all N-1 peers.
Under NUMA:

- No direct GPU-GPU link -- every message bounces through CPU memory
- Simultaneous N-way fan-out saturates PCIe upstream bandwidth
- Cross-socket messages compete for UPI bandwidth (~64-100 GB/s shared)

**At N=8, One-Shot under NUMA is not 1-step -- it serializes on PCIe.**
Ring Allreduce with topology-aware construction is the correct algorithm here.

## Topology-Aware Ring Construction

The optimal ring for a 2-socket 4-GPU-per-socket node minimizes UPI crossings to exactly 2:

```
Ring order:  GPU0 → GPU1 → GPU2 → GPU3 → GPU4 → GPU5 → GPU6 → GPU7
             └──── socket0 ────┘  ↑UPI↑  └──── socket1 ────┘
                                  └─────────────────────────────┘ (wrap)
```

This means 6 of 8 hops stay on-socket (fast PCIe), only 2 cross UPI.
A naively ordered ring (0,4,1,5,2,6,3,7) would alternate sockets on every hop -- 8x the UPI traffic.

The difference in hop cost:

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
# Correct: ranks 0-3 on socket 0, ranks 4-7 on socket 1
I_MPI_PIN_DOMAIN=socket mpirun -n 8 -ppn 8 python train.py

# Verify ring construction
CCL_LOG_LEVEL=info python script.py 2>&1 | grep -i "ring order"
```

## Algorithm Decision Tree for CRI Inference

```
What collective do you need?
│
├── Allreduce (TP layer sync)
│     ├── msg < 1MB, TP <= 8  →  Ring              ✓ Done
│     └── msg > 1MB           →  Ring (pipelined)  ✓ Done
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
| 64 KB -- 4 MB | Ring (pipelined RS+AG) | Pipelined reduce-scatter + allgather fills PCIe bandwidth |
| > 4 MB | Ring (large-message path) | Bandwidth-bound, ring optimal |
| Any | NOT One-Shot | Fan-out saturates PCIe |

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
