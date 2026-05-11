# oneCCL Inference Tutorial

Internal tutorial covering oneCCL collective communications for inference workloads
on CRI hardware (NUMA-only topology).

## Build the Site

```bash
# Install dependencies
pip install -r requirements.txt

# Build
jupyter-book build .

# View locally
open _build/html/index.html

# Deploy to internal web server
rsync -av _build/html/ your-intel-server:/path/to/docs/oneccl/
```

## Run Notebooks Interactively

```bash
module load intel/mpi
pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Launch JupyterLab
jupyter lab
```

Each notebook uses the MPI launcher pattern -- run notebook cells normally from a
single-rank kernel; the cells invoke `mpirun` internally.

## Structure

```
oneccl-tutorial/
├── _config.yml              # Jupyter Book config
├── _toc.yml                 # Table of contents
├── intro.md                 # Landing page
├── chapters/
│   ├── 00_foundations.md    # From first principles: ranks, ring, recursive doubling
│   ├── 01_overview.md       # oneCCL API and stack position
│   ├── 02_topology.md       # NUMA topology + algorithm selection
│   ├── 03_when_to_use.md    # Decision guide: which collective for which workload
│   ├── nixl_boundary.md     # oneCCL vs NIXL boundary
│   └── perf_tuning.md       # CCL_* and I_MPI_* env var reference
└── notebooks/
    ├── 02_environment_setup.ipynb
    ├── 03a_allreduce_walkthrough.ipynb   # Core: TP decode hot path
    ├── 03b_allgather.ipynb               # Sequence parallelism KV gather
    ├── 03c_alltoall_moe.ipynb            # MoE expert routing (alltoallv)
    └── 04_inference_tp_decode.ipynb      # End-to-end TP decode loop
```

## Hardware Assumptions

All content calibrated for **CRI nodes: NUMA-only, no UALink/XeLink**.
Algorithm selection (Ring preferred, One-Shot deprioritized) reflects this topology.
