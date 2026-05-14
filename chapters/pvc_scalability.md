# PVC Host-Staged Scaleout in oneCCL

## Abstract

This chapter argues that oneCCL's multi-node GPU path on Intel Ponte Vecchio
(PVC) on Aurora is limited by host-staged scaleout. HMEM is not a usable
alternative: the Slingshot-11 CXI libfabric provider has no `FI_HMEM_ZE`
backend for Intel GPU memory, making host staging the only functional
inter-node path on Aurora today. The
argument rests on oneCCL source code. The relevant paths show five points:
(1) the SYCL+ZE allreduce path selects `topo` as the main GPU algorithm,
(2) small BF16 scaleout messages in the decode regime always select the `direct` scaleout path,
(3) the `topo` scaleout path stages through host buffers when HMEM is disabled,
(4) the SYCL scaleout implementation runs a sequential
`D2H -> host_task(allreduce+wait) -> H2D` chain, and (5) the OFI transport
forces `copy_to_host = true` inside that SYCL scaleout path. The result is a
fixed host bounce buffer term on every inter-node step of a latency-sensitive
collective. On Aurora, Xe Link reduces intra-node cost but does not remove this
inter-node term. This chapter does not infer a universal node-count wall or a
specific MPI algorithm where oneCCL source does not establish one.

---

## 1. Introduction

This chapter uses Aurora as the reference PVC deployment. Recent Aurora system
papers describe a 10,624-node machine in which each node carries two Intel
Xeon Max CPUs and six Intel Data Center Max GPUs (PVC), with the GPUs
connected in an all-to-all Xe Link topology inside the node and Slingshot-11
used for inter-node scaleout. That matters here because it removes ambiguity
about the hardware baseline: Aurora already provides a strong intra-node GPU
fabric and a large production scaleout network.

Those same Aurora papers also make the hardware-software boundary clearer. At
the platform level, Aurora is not defined by an absence of GPU-to-network
capability. The narrower question for this chapter is which oneCCL software path
is actually selected for PVC collectives. Within a node, Xe Link provides high-
bandwidth GPU communication. Across nodes, the software stack still determines
whether GPU data reaches the NIC directly or passes through host memory.

For oneCCL, that distinction matters most in decode-time tensor parallelism,
where allreduce messages are small and latency-sensitive. In this regime,
inter-node collective performance is dominated less by peak bandwidth than by
the fixed cost imposed on each step of the scaleout path.

This chapter focuses on the software mechanism behind that fixed cost: the host
bounce buffer. It does not estimate a node-count crossover. It confirms, from
source code, that the default PVC scaleout path inserts host staging into
inter-node GPU collectives and explains why that limits scaling.

---

## 2. Scope and Claim

This chapter is limited to:

- Intel Ponte Vecchio (PVC, Xe-HPC)
- GPU-resident buffers
- SYCL+ZE oneCCL builds
- multi-node scaleout
- latency-sensitive allreduce, especially decode-time tensor parallelism

This chapter does not attempt to cover:

- training gradient synchronization
- MoE alltoall behavior
- a benchmark-derived node-count wall

For DeepSeek R1, V3, and V4 work planned on PVC, read this chapter as the
baseline allreduce note. It covers the host-staged inter-node path used by
oneCCL. It does not try to model MoE expert-routing traffic.

The central claim is:

> On PVC on Aurora, oneCCL's multi-node GPU allreduce path is host-staged on
> scaleout — this is not the default among several options but the only
> functional path, because the CXI provider does not support `FI_HMEM_ZE` for
> Intel GPU memory. That host staging adds a fixed software latency term to
> every inter-node step of the collective.

The PVC scaling limit for small-message decode is therefore not set by lack of
intra-node bandwidth. It is set by repeating a host-mediated inter-node path
as node count grows, with no software knob currently able to bypass it on
Aurora.

---

## 3. Why PVC Still Has an Inter-Node Software Problem

PVC has Xe Link. That reduces the cost of local reduce-scatter and allgather
work inside `topo`. It does not, by itself, solve the inter-node scaleout path.

Conceptually, the collective splits into two parts:

```text
intra-node on PVC:     GPU/tile <-> Xe Link <-> GPU/tile
inter-node scaleout:   GPU/tile -> host software path -> network -> remote node
```

For small decode-time messages, the inter-node path dominates because:

- the message is too small to amortize fixed orchestration overhead well
- the allreduce result is on the critical path of the next layer
- local Xe Link efficiency cannot remove a repeated inter-node staging penalty

The rest of this chapter shows that this penalty is encoded in the selected
oneCCL path.

---

## 4. Source-Code Confirmation

### 4.1 GPU Allreduce Enters `topo`

In the SYCL+ZE configuration, oneCCL installs `topo` as the main allreduce
algorithm:

```cpp
insert(main_table, 0, CCL_SELECTION_MAX_COLL_SIZE, ccl_coll_allreduce_topo);
```

Source: `src/coll/selection/selector_allreduce.cpp`

This is the correct starting point for PVC analysis because it is the GPU path
that oneCCL actually selects. The question is not whether oneCCL contains ring
or recursive-doubling code elsewhere. The question is what `topo` does at
multi-node scaleout.

### 4.2 Small BF16 Scaleout Messages Select `direct`

For the SYCL scaleout auto-selector, BF16 messages remain on the `direct`
scaleout path for relatively small sizes:

```cpp
else if (ccl_dtype == ccl::datatype::bfloat16) {
    if (comm_size <= 8 && size <= 1024 * 1024) {
        return { allreduce_scaleout_algo::direct };
    }
    if (comm_size > 8 && size <= 4 * 1024 * 1024) {
        return { allreduce_scaleout_algo::direct };
    }
}
```

Source: `src/coll/algorithms/utils/sycl_selection.cpp`

For any multi-node PVC deployment at ≥2 nodes with 12 tiles/node, `comm_size`
is ≥24, so the `comm_size > 8` branch is always active. The effective threshold
for Aurora-scale decode is `size <= 4 MB`. A batch-1, `d_model = 8192`, BF16
activation is 16 KB — 250× below that threshold. Decode-time TP activations
unconditionally select `direct` within this scope, where fixed step cost
dominates.

### 4.3 `topo` Scaleout Uses Host Buffers When HMEM Is Disabled

The `topo` scaleout path computes an `enable_hmem` gate and explicitly enters a
host-buffer path when the gate is false:

```cpp
bool enable_hmem = (ccl::global_data::env().use_hmem && atl_base_comm::attr.out.enable_hmem);

if (multi_node) {
    if (!enable_hmem) {
        LOG_DEBUG("topo/scale_out: use host_", ccl_coll_type_to_str(coll_param.ctype));
        out_event = fill_scaleout_coll_param(in_coll_param, coll_param, sched, wait_events);
        sched->add_barrier();
    }
    coll_param.is_scaleout = true;
    coll_param.is_hmem_enabled = enable_hmem;
    out_event = ccl::add_coll(sched, coll_param, wait_events);
}
```

Source: `src/coll/coll_util.cpp`

This is the first direct confirmation of the host bounce buffer mechanism. The
multi-node GPU path branches on `!enable_hmem` and, in that case, prepares
scaleout parameters using host-side buffers before entering the collective.
Later logic decides whether an H2D copy is needed after scaleout. Together,
these branches show that the default multi-node GPU path is staged through host
memory when HMEM is not active.

### 4.4 The SYCL Scaleout Implementation Is Sequential

The direct scaleout implementation in `allreduce_scaleout_sycl.cpp` builds the
path in three phases.

First, if `copy_to_host` is true, it copies from the device buffer to a host
staging buffer:

```cpp
if (copy_to_host) {
    scaleout_send_buf = MPI_IN_PLACE;
    scaleout_recv_buf = comm->get_scaleout_host_buf();
    copy_e = q.submit([=](sycl::handler& h) {
        h.depends_on(dep_events);
        h.memcpy(scaleout_recv_buf,
                 send_buf == MPI_IN_PLACE ? recv_buf : send_buf,
                 count * ccl_dtype.size());
    });
    dep_events.clear();
    dep_events.push_back(std::move(copy_e));
}
```

Next, it runs the collective inside a host task and waits for completion:

```cpp
op_end = q.submit([=](sycl::handler& h) {
    h.depends_on(dep_events);
    h.host_task([=]() {
        int ep_idx = 0;
        atl_req_t req;
        ATL_CALL_THROW_IF_ERROR(atl_comm->allreduce(ep_idx,
                                                    scaleout_send_buf,
                                                    scaleout_recv_buf,
                                                    count,
                                                    ccl_dtype.atl_datatype(),
                                                    static_cast<atl_reduction_t>(reduction),
                                                    req));

        ATL_CALL_THROW_IF_ERROR(atl_comm->check(ep_idx, req));
        if (!req.is_completed) {
            ATL_CALL_THROW_IF_ERROR(atl_comm->wait(ep_idx, req));
        }
    });
});
```

Finally, if `copy_to_host` is true, it copies the completed result back to the
device buffer:

```cpp
if (copy_to_host) {
    op_end = q.submit([=](sycl::handler& h) {
        h.depends_on(op_end);
        h.memcpy(recv_buf, scaleout_recv_buf, count * ccl_dtype.size());
    });
}
```

Source: `src/coll/algorithms/allreduce/sycl/allreduce_scaleout_sycl.cpp`

This is the core confirmation of the host bounce buffer. The path is an
explicit event chain:

```text
D2H memcpy -> host_task(allreduce + wait) -> H2D memcpy
```

The source also contains the warning:

```cpp
"Falling back. TODO: chunking/pipelining"
```

That warning matters because a staged path could, in principle, recover some
latency through chunking and overlap. This implementation states that such
pipelining is not present here.

### 4.5 OFI Forces `copy_to_host` in This SYCL Path

The same file makes the transport coupling explicit:

```cpp
bool copy_to_host = ccl::global_data::env().sycl_enable_direct_gpu_rdma ? false : true;
...
if (should_disable_rdma(ze_dev) || ccl::global_data::env().atl_transport == ccl_atl_ofi) {
    copy_to_host = true;
}
```

Source: `src/coll/algorithms/allreduce/sycl/allreduce_scaleout_sycl.cpp`

This supports a narrow OFI claim: in this SYCL allreduce scaleout path, OFI
forces `copy_to_host = true`. It does not prove that every OFI-related code path
in the library stages through host memory. It does prove the mechanism relevant
to the PVC decode argument.

---

## 5. What the Source Establishes, and What It Does Not

The claim should stay inside the source evidence boundary.

| Statement | Established from oneCCL source? | Reason |
|---|---|---|
| `topo` is the main SYCL+ZE GPU allreduce path | Yes | `selector_allreduce.cpp` |
| small BF16 scaleout messages often select `direct` | Yes | `sycl_selection.cpp` |
| `!enable_hmem` causes host staging in `topo` scaleout | Yes | `coll_util.cpp` |
| the SYCL direct scaleout path is `D2H -> host_task -> H2D` | Yes | `allreduce_scaleout_sycl.cpp` |
| OFI forces `copy_to_host` inside that SYCL path | Yes | `allreduce_scaleout_sycl.cpp` |
| the exact MPI algorithm used after `direct` delegation | No | oneCCL delegates; MPI must be inspected or measured separately |
| a universal PVC node-count wall | No | requires workload-specific measurement |
| node-wide serialization through a single ATL endpoint | No | `ep_idx = 0` is visible here, but the broader system claim needs more evidence |

The source supports a narrower claim than "oneCCL proves recursive doubling on
Aurora":

- oneCCL source proves the presence of a host-staged inter-node path in the
  relevant default baseline
- that staged path is sequential and host-orchestrated
- therefore it adds a repeated fixed software cost to scaleout

That is enough to explain why host bounce buffering is a software scalability
limit even without site-specific benchmark numbers.

---

## 6. Why Host Bounce Buffering Limits PVC Decode Scaling

For decode-time tensor parallelism, a useful per-layer abstraction is:

```text
T_layer(N) = T_compute(N) + 2 * T_allreduce(N)
```

The allreduce cost can be decomposed conceptually as:

```text
T_allreduce(N) = T_intra_pvc + T_scaleout(N)
```

On PVC, `T_intra_pvc` benefits from Xe Link. The source-confirmed software issue
is in `T_scaleout(N)`.

The host-staged path implies a fixed per-step coefficient of the form:

```text
alpha_staged = D2H + host scheduling/posting + collective progress/wait + H2D
```

For small messages, the bandwidth term is small enough that this fixed
coefficient dominates the scaleout cost. As node count grows, any collective
that requires additional inter-node steps multiplies that coefficient. At the
same time, tensor-parallel compute per rank shrinks with scale.

The consequence is straightforward:

- Xe Link improves the local phase but leaves the staged inter-node coefficient
  intact
- small-message decode cannot amortize the staged coefficient well
- increasing node count reduces compute per rank faster than it reduces the
  software cost of each inter-node collective step

That is the software scalability limitation due to host bounce buffering.

This chapter does not need benchmark numbers to make the mechanism clear. The
source already shows the repeated staged path. Measurements are only needed to
determine where it becomes dominant for a given workload.

For DeepSeek R1, V3, and V4 on PVC, this conclusion still matters anywhere
oneCCL allreduce remains on the critical path. It is not the full DeepSeek
communication story. Expert-routing and other alltoall-heavy paths need a
separate PVC analysis.

---

## 7. HMEM Is Not Available on Aurora

The source makes clear that HMEM is gated:

```cpp
bool enable_hmem = (ccl::global_data::env().use_hmem && atl_base_comm::attr.out.enable_hmem);
```

`atl_base_comm::attr.out.enable_hmem` is set during communicator init by probing
the libfabric provider for `FI_HMEM` support. For Intel GPU memory, oneCCL
registers buffers with `fi_mr_regattr(iface=FI_HMEM_ZE)` — the Intel Level Zero
HMEM interface. The CXI provider (Slingshot-11) has no `FI_HMEM_ZE` backend:
its `prov/cxi/src/` implements HMEM only for CUDA (`disable_dmabuf_cuda`) and
ROCm (`disable_dmabuf_rocr`).

Two failure modes on Aurora:

1. The CXI probe with `FI_HMEM` hints fails → `LOG_WARN` → `enable_hmem` stays
   false → oneCCL silently continues with host staging. `CCL_ATL_HMEM=1`
   appears to be accepted but does nothing.

2. The probe passes (CXI advertises `FI_HMEM` generically when requested), but
   the first `fi_mr_regattr` with `iface=FI_HMEM_ZE` fails → `CCL_THROW` →
   fatal exception at first collective.

Either way, `CCL_ATL_HMEM=1` is not a usable knob on Aurora today. The baseline
host-staged path described in this note is not a choice — it is the only
functional path. HMEM becomes available only when the CXI provider adds an Intel
Level Zero GPU memory backend.

---

## 8. Conclusion

The oneCCL source code supports a precise PVC claim.

For SYCL+ZE GPU allreduce on PVC on Aurora, the library selects `topo` for the
main GPU path, enters a small-message `direct` scaleout path for typical decode
activations, stages through host memory, and executes a sequential
`D2H -> host_task(allreduce + wait) -> H2D` chain in that scaleout path. Under
OFI, that path forces `copy_to_host = true`. The HMEM bypass is not available
on Aurora because the CXI libfabric provider has no `FI_HMEM_ZE` backend for
Intel GPU memory — host staging is the only functional inter-node path today.

Host bounce buffering is not an incidental implementation detail. It is the
only inter-node software path for PVC on Aurora. Because it inserts a fixed
host-mediated cost into every inter-node step, it limits oneCCL scalability
on PVC for small, latency-sensitive decode collectives even though PVC has
strong intra-node Xe Link bandwidth.

The remaining work for a system-specific study is to measure the workload-
specific point at which that fixed coefficient becomes the dominant term in
end-to-end decode latency.

---

## References

### oneCCL source

- `src/coll/selection/selector_allreduce.cpp`
- `src/coll/algorithms/utils/sycl_selection.cpp`
- `src/coll/coll_util.cpp`
- `src/coll/algorithms/allreduce/sycl/allreduce_scaleout_sycl.cpp`

### Aurora and measurement references

- Allcock et al., "Aurora: Architecting Argonne's First Exascale
    Supercomputer for Accelerated Scientific Discovery" (arXiv:2509.08207)
  — ECB topology, PCIe Gen5 x16 GPU→CPU (64 GB/s), PCIe Gen4 NIC path (32 GB/s)
- Ibeid et al., "Scaling MPI Applications on Aurora" (arXiv:2512.04291)
  — allreduce latency scaling data, NIC effective bandwidth, PCIe Gen4→Gen5
    conversion overhead
- Goto et al., "Sustaining Exascale Performance: Lessons from HPL and
    HPL-MxP on Aurora" (arXiv:2604.09517)
  — confirms PCIe switches fan out Gen5 x16 to Gen4 x16 for NIC-facing ports
- Hidayetoglu et al., "CommBench: Micro-benchmarking Hierarchical Networks
    with Multi-GPU, Multi-NIC Nodes" (ICS 2024)
  — ~8 µs allreduce latency on Aurora with (n, 12, 12) tile configuration;
    direct empirical support for the staged per-step coefficient in §6