# oneCCL Inference Tutorial

Internal tutorial covering oneCCL collective communications for inference and training
workloads on Intel Xe GPUs (BMG/CRI and PVC).

## Build the Site

```bash
pip install -r requirements.txt
jupyter-book build .
open _build/html/index.html
```

## Run Notebooks Interactively

```bash
module load intel/mpi
pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
jupyter lab
```

Each notebook uses the MPI launcher pattern — run notebook cells normally from a
single-rank kernel; the cells invoke `mpirun` internally.

## Structure

```
oneccl-tutorial/
├── intro.md                 # Landing page and navigation
├── chapters/
│   ├── 00_foundations.md          # From first principles: collectives, alpha-beta model, algorithms
│   ├── 01_overview.md             # oneCCL API and stack position
│   ├── 02_topology.md             # NUMA topology and algorithm selection
│   ├── 03_when_to_use.md          # Decision guide: which collective for which workload
│   ├── pvc_hardware_constraints.md # GPU hardware limits (BMG/CRI + PVC context)
│   ├── nixl_boundary.md           # oneCCL vs NIXL boundary for disaggregated inference
│   ├── perf_tuning.md             # CCL_* and I_MPI_* env var reference
│   ├── debugging_profiling.md     # Diagnostics, VTune, log interpretation
│   ├── native_cpp_api.md          # Native C++ / SYCL integration (no PyTorch)
│   ├── pvc_scalability.md         # PVC host-staged scaleout (Aurora, source analysis)
│   └── host_staging_scaling.md    # Host staging scaling wall (model + evidence)
└── notebooks/
    ├── 02_environment_setup.ipynb
    ├── 03a_allreduce_walkthrough.ipynb   # TP decode hot path
    ├── 03b_allgather.ipynb               # Sequence parallelism KV gather
    ├── 03c_alltoall_moe.ipynb            # MoE expert routing (alltoallv)
    └── 04_inference_tp_decode.ipynb      # End-to-end TP decode loop
```

## Hardware Assumptions

Primary target: **BMG/CRI (Xe3, NUMA-only, no XeLink)**. PVC (Xe-HPC, Aurora) is
covered in the scaleout chapters. Both share host-staged inter-node defaults but
differ on intra-node fabric (PVC has Xe Link, BMG/CRI has only PCIe).
