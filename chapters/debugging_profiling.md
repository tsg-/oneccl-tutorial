# Debugging and Profiling oneCCL

This chapter covers the "my collective hung" and "my collective is slow" scenarios. It maps
symptoms to diagnostic steps, shows what the log output looks like, and explains how to connect
oneCCL to Intel VTune for timeline profiling.

---

## Triage: Is It a Hang or Slowness?

```
Symptom: process is stuck, no output for > 30 seconds
│
├── All ranks stuck at the same collective call
│   └── Likely cause: one rank never reached the call
│       → Check for asymmetric control flow (see §1)
│
├── Some ranks return, others stuck
│   └── Likely cause: mismatched counts, wrong communicator, or wrong rank
│       → Check CCL_LOG_LEVEL=debug output for count mismatch (see §2)
│
├── Process hangs on first init, never prints rank
│   └── Likely cause: KVS bootstrap failure (MPI not started, wrong transport)
│       → Check MPI is initialized, CCL_ATL_TRANSPORT matches your setup (see §3)
│
└── Process hangs during scale-out phase only (multi-node)
    └── Likely cause: OFI provider mismatch, firewall, or HMEM silent hang
        → Check I_MPI_FABRICS, OFI_PROVIDER; disable CCL_ATL_HMEM if set (see §4)

Symptom: collective completes but is slow (> 2× expected latency)
│
├── Single-node: slow intra-node phase
│   └── NUMA pinning wrong, P2P fallback to host staging
│       → Verify with CCL_LOG_LEVEL=info, check "p2p" / "ipc" / "host_buf" in output
│
└── Multi-node: slow inter-node phase
    └── Host staging overhead, wrong scaleout algorithm, NIC not used
        → CCL_LOG_LEVEL=info, check "scaleout" entries; measure with VTune (see §5)
```

---

## §1: Asymmetric Control Flow (Most Common Hang)

Every collective is a **barrier**: all ranks in the communicator must call it. If any rank
skips a collective — due to an `if` branch, an early return, or an exception — every other
rank will wait forever.

```python
# WRONG: rank 0 skips the collective when processing the first batch
if rank == 0 and step == 0:
    print("skipping warmup allreduce")
else:
    dist.all_reduce(tensor)      # ranks 1-3 hang here forever

# RIGHT: all ranks participate, even if the result isn't used
dist.all_reduce(tensor)          # always called on all ranks
if rank == 0 and step == 0:
    print("warmup done")
```

In practice this shows up as:
- Error handling that only one rank catches (rank 0 loads a bad checkpoint, raises exception,
  skips the following broadcast; other ranks wait on broadcast forever)
- Conditional gradient syncs in custom training loops

**Diagnosis:** add a `dist.barrier()` before the suspected collective. If the hang moves to
the barrier, you have asymmetric control flow.

---

## §2: Mismatched Counts

`alltoallv` requires `send_counts[j]` from rank i to equal `recv_counts[i]` from rank j. If
they don't match, the behavior is undefined — it may hang, return wrong data silently, or
corrupt memory.

```python
# WRONG: send side sends 10 tokens to rank 1, but recv side expects 8
# rank 0:
send_counts = [0, 10, 5, 3]      # sends 10 to rank 1
# rank 1:
recv_counts = [8, 0, 4, 6]       # expects only 8 from rank 0  ← mismatch
```

**Diagnosis script:**

```python
import torch
import torch.distributed as dist

def verify_alltoallv_counts(send_counts, world_size, rank):
    """
    Send counts must satisfy: send_counts[j] on rank i == recv_counts[i] on rank j.
    This function allgathers all send_counts and checks the transpose consistency.
    """
    send_t = torch.tensor(send_counts, dtype=torch.int64)
    all_counts = [torch.zeros(world_size, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(all_counts, send_t)   # all_counts[i][j] = rank i's send_counts[j]

    # Check: all_counts[i][rank] should equal all_counts[rank][i] for all i
    my_expected_recv = [all_counts[i][rank].item() for i in range(world_size)]
    for i in range(world_size):
        if my_expected_recv[i] != send_counts[i]:  # this rank's send to i
            print(f"rank {rank}: MISMATCH rank {i}: "
                  f"they send me {my_expected_recv[i]}, I send them {send_counts[i]}")
    return my_expected_recv
```

---

## §3: Log Level Diagnostics

oneCCL's logging system is controlled by `CCL_LOG_LEVEL`. Set it **before** `init_process_group`:

```bash
# warn (default): only errors and warnings
export CCL_LOG_LEVEL=warn

# info: algorithm selection, ring construction, schedule summary
export CCL_LOG_LEVEL=info

# debug: schedule entry execution, buffer sizes, transport decisions
export CCL_LOG_LEVEL=debug

# trace: everything, including per-step timing inside the progress engine
export CCL_LOG_LEVEL=trace
```

### What to look for at `info` level

```bash
CCL_LOG_LEVEL=info mpirun -n 4 python script.py 2>&1 | grep -E "ring|topo|p2p|ipc|scale"
```

Expected healthy output for a 4-GPU BMG/CRI node:

```
[CCL][info] topo_manager: p2p access: GPU0<->GPU1 OK (PCIe), GPU0<->GPU2 OK (UPI), ...
[CCL][info] ring_allreduce: ring order: 0 1 2 3 (topology-aware)
[CCL][info] topo/allreduce: scale_up via ze_ipc, scale_out via ofi host_buf
```

Unhealthy — P2P fell back to host staging:

```
[CCL][info] topo_manager: p2p access: GPU0<->GPU1 FAILED (iommu or ACS blocked)
[CCL][info] topo/allreduce: scale_up via host_buf (p2p unavailable)
```

Check this line first. If `scale_up via host_buf` appears, you are paying
host-staging cost for intra-node communication. Expected intra-node bandwidth drops from
~25 GB/s (PCIe P2P) to ~5-10 GB/s.

### What to look for at `debug` level

```bash
CCL_LOG_LEVEL=debug mpirun -n 4 python script.py 2>&1 | grep -E "allreduce|count|dtype|algo" | head -30
```

```
[CCL][debug] allreduce: count=8192 dtype=bfloat16 algo=topo
[CCL][debug] allreduce: scaleout_algo=ring transport=ofi copy_to_host=true
[CCL][debug] entry: ze_copy (GPU->host) size=16384 bytes
[CCL][debug] entry: ofi_allreduce on host_buf size=16384 bytes
[CCL][debug] entry: ze_copy (host->GPU) size=16384 bytes
```

The `copy_to_host=true` line confirms you are on the default host-staged path. If
`copy_to_host=false` appears, HMEM is active.

---

## §4: Init and Transport Failures

### Collective hangs at first call, init appears to succeed

```bash
# Step 1: verify all ranks actually initialized
mpirun -n 4 python -c "
import torch.distributed as dist
import oneccl_bindings_for_pytorch
dist.init_process_group(backend='ccl')
print(f'rank {dist.get_rank()}/{dist.get_world_size()} OK')
dist.destroy_process_group()
"
# Expected: four lines "rank 0/4 OK" through "rank 3/4 OK"
# If fewer than 4 lines: one rank crashed during init
```

### Wrong transport

```bash
# Check which transport oneCCL actually negotiated
CCL_LOG_LEVEL=info mpirun -n 4 python script.py 2>&1 | grep -i "transport\|ofi\|mpi\|atl"
```

For multi-node, `ofi` is required. If you see `mpi` transport but your MPI installation
doesn't have OFI support, switch:

```bash
export CCL_ATL_TRANSPORT=ofi
export I_MPI_FABRICS=shm:ofi
export I_MPI_OFI_PROVIDER=verbs   # or psm3, depending on your NIC
```

### HMEM silent hang

If you have `CCL_ATL_HMEM=1` set and the collective hangs without error output, HMEM failed
to negotiate `FI_HMEM` with the provider. Disable it:

```bash
unset CCL_ATL_HMEM
# or explicitly:
export CCL_ATL_HMEM=0
```

---

## §5: VTune Integration

oneCCL emits Intel ITT (Instrumentation and Tracing Technology) markers that appear as named
tasks in VTune's Timeline view. These let you see exactly when the Progress Engine is blocked
waiting on a Level Zero IPC fence, or when the OFI send is posted vs. completed.

### Build requirement

ITT markers are only emitted if oneCCL was built with `ONECCL_ENABLE_ITT` defined. The
internal wrapper is at `src/internal/itt_wrapper.hpp`:

```cpp
// oneCCL wraps ITT tasks like this:
namespace itt {
class Task {
    void start() { __itt_task_begin(domain_, ...); }
    void end()   { __itt_task_end(domain_); }
};
}
```

The ITT domain name is `"oneCCL2"` — use this to filter in VTune.

### Collecting a VTune trace

```bash
# Enable ITT markers
export CCL_ITT_LEVEL=1

# Collect with VTune hotspots + GPU offload analysis
vtune -collect gpu-offload \
      -knob enable-stack-collection=true \
      -- mpirun -n 4 python your_inference_script.py

# Or: collect threading analysis to see Progress Engine thread behavior
vtune -collect threading \
      -- mpirun -n 4 python your_inference_script.py
```

### What to look for in the timeline

```
Timeline (zoomed to one allreduce, ~30 µs window):

  GPU compute queue:  ████████ GEMM ████████
  CPU thread:                                 [CCL worker: ze_copy D2H] [ofi_send] [ofi_wait] [ze_copy H2D]
  GPU CCL queue:                              [D2H DMA]                            [H2D DMA]
                                              ▲                                    ▲
                               oneCCL2/ze_copy_start              oneCCL2/ofi_complete
```

The ITT task markers appear as colored bands in VTune's Thread column under the CPU thread that
runs the oneCCL progress engine (usually worker thread 0). The names follow the pattern
`oneCCL2/<entry_type>`.

Key markers to check:
- `oneCCL2/ze_copy` — D2H and H2D PCIe DMA. Should be ~2 µs each. If >5 µs: PCIe contention
  or wrong copy engine selected.
- `oneCCL2/ofi_*` — network phase. The gap between `ofi_send` and `ofi_complete` is wire
  latency + host-staging overhead. Should be ~5-8 µs for 16 KB on a 100 GbE link.
- Long CPU spin inside `ofi_wait` without progress: OFI provider is stuck or NIC is saturated
  by another traffic class (check for KV transfers on the same NIC queue).

### Python profiling alternative (no VTune required)

For a quick latency breakdown without VTune:

```python
import time
import torch
import torch.distributed as dist

# Warm up
for _ in range(10):
    dist.all_reduce(tensor)
torch.xpu.synchronize()

# Measure
times = []
for _ in range(100):
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    dist.all_reduce(tensor)
    torch.xpu.synchronize()   # waits for collective + all GPU work to finish
    times.append(time.perf_counter() - t0)

times.sort()
p50 = times[50] * 1e6
p95 = times[95] * 1e6
p99 = times[99] * 1e6
if dist.get_rank() == 0:
    print(f"allreduce latency: p50={p50:.1f} µs  p95={p95:.1f} µs  p99={p99:.1f} µs")
```

High p99/p50 ratio (> 3×) usually means:
- NUMA misalignment causing occasional cross-socket UPI spikes
- Background processes competing on PCIe (check `cat /proc/interrupts`)
- OFI provider retransmit on lossy fabric

---

## §6: Collective Correctness Verification

Before debugging performance, verify correctness. Wrong results are harder to spot than hangs.

```python
import torch
import torch.distributed as dist
import oneccl_bindings_for_pytorch

dist.init_process_group(backend="ccl")
rank = dist.get_rank()
world = dist.get_world_size()

# Allreduce correctness: each rank contributes rank+1, expected sum = world*(world+1)/2
N = 8192
tensor = torch.full((N,), float(rank + 1), dtype=torch.bfloat16).to("xpu")
dist.all_reduce(tensor)

expected = world * (world + 1) / 2
# BF16 has limited precision — use atol appropriate for accumulation
if not torch.allclose(tensor, torch.full((N,), expected, dtype=torch.bfloat16).to("xpu"),
                      atol=0.1 * world):
    print(f"rank {rank}: WRONG allreduce result! got {tensor[0].item()}, expected {expected}")
else:
    if rank == 0:
        print(f"allreduce CORRECT: {tensor[0].item()} == {expected}")

dist.destroy_process_group()
```

Run this after any environment change (new driver, new oneCCL version, new NUMA pinning) before
putting traffic on the system.

---

## Common Issues Quick Reference

| Symptom | Likely cause | Fix |
|---|---|---|
| All ranks hang at allreduce | One rank skipped the call | Add `dist.barrier()` before to confirm; fix asymmetric code |
| Alltoallv silently wrong data | `send_counts`/`recv_counts` mismatch | Run `verify_alltoallv_counts()` diagnostic |
| Init succeeds but first collective hangs | Transport not available (OFI not configured) | Check `CCL_ATL_TRANSPORT`, `I_MPI_FABRICS`, `I_MPI_OFI_PROVIDER` |
| Hang only on multi-node, single-node fine | HMEM enabled and NIC can't negotiate FI_HMEM | Unset `CCL_ATL_HMEM` |
| Collective takes 3-5× expected time | P2P fell back to host staging (intra-node) | Check `CCL_LOG_LEVEL=info` for "host_buf"; fix IOMMU/ACS/permissions |
| High p99 latency, p50 normal | NUMA misalignment or competing traffic | Verify `I_MPI_PIN_DOMAIN=socket`; check `/proc/interrupts` |
| `scale_up via host_buf` in logs | P2P access check failed at init | Check `/dev/dri/renderD*` permissions, IOMMU mode, ACS settings |
| Broadcast hangs at startup | Root rank hasn't loaded data yet when others call | Ensure rank 0 finishes loading before all ranks call broadcast |
