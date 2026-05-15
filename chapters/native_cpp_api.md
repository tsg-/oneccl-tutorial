# Native C++ / SYCL Integration

The notebooks in this tutorial use `torch.distributed` with the CCL backend — the Python
path that most inference and training engineers reach first. This chapter covers the **native
C++ API** directly.

**Use the native API when:**
- Building a C++ inference engine (vLLM-style engine written in C++ / SYCL)
- Embedding oneCCL into an existing SYCL compute pipeline without a PyTorch dependency
- Benchmarking without Python overhead
- Debugging by reading what the library actually does rather than going through bindings

---

## API Families

oneCCL exposes two API surfaces. Both are shipped in `oneapi/ccl.h`:

| API | Style | Header guard | Use when |
|---|---|---|---|
| **v2 C API** (`oneccl*`) | C-style handles, `onecclComm_t`, function pointers | None | C++ programs, vLLM-style engines, SYCL integration |
| **v1 C++ template API** (`ccl::*`) | C++ classes, `ccl::communicator`, `ccl::event` | `CCL_PRODUCT_FULL` | Generic C++ programs, non-SYCL paths |

The examples in `oneCCL/examples/sycl/sycl.cpp` use the v2 C API. **This chapter uses the
v2 C API throughout** — it is the API path supported for SYCL queue integration and it is
what you will find in Intel's published examples and the `oneccl_bindings_for_pytorch` internals.

---

## Minimal SYCL Allreduce

The complete pattern for an allreduce on a SYCL device buffer:

```cpp
// Build: icpx -fsycl -o example example.cpp -lmpi -loneccl
// Run:   mpirun -n 4 ./example
//
// Source references:
//   oneCCL/examples/sycl/sycl.cpp  — two-queue overlap pattern (this file)
//   oneCCL/examples/simple/simple.cpp  — minimal single-queue version

#include "oneapi/ccl.h"
#include <mpi.h>
#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    // 1. Initialize MPI (oneCCL uses MPI for rank/size discovery)
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, nullptr);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 2. Distribute a unique ID so all ranks connect to the same communicator.
    //    Rank 0 generates it, all ranks receive it via MPI_Bcast (out-of-band).
    onecclUniqueId uid;
    if (rank == 0) onecclGetUniqueId(&uid);
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);

    // 3. Select the GPU for this rank (local_rank = rank % gpus_per_node)
    //    onecclSetDevice tells oneCCL which Level Zero device to use for P2P IPC handles.
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    onecclSetDevice(local_rank);

    // 4. Create the oneCCL communicator, wired up via the shared uid.
    onecclComm_t comm;
    onecclCommInitRank(&comm, world_size, uid, rank);

    // 5. Create a SYCL queue bound to this rank's GPU (Level Zero backend).
    auto platforms = sycl::platform::get_platforms();
    sycl::queue q;
    for (auto& p : platforms) {
        if (p.get_backend() == sycl::backend::ext_oneapi_level_zero) {
            auto devs = p.get_devices();
            q = sycl::queue(devs[local_rank % devs.size()],
                            {sycl::property::queue::in_order{}});
            break;
        }
    }

    // 6. Allocate device memory and initialize.
    constexpr size_t N = 8192;            // 16 KB of float16 ≈ one TP allreduce
    auto* send = sycl::malloc_device<uint16_t>(N, q);  // BF16
    auto* recv = sycl::malloc_device<uint16_t>(N, q);
    q.fill(send, static_cast<uint16_t>(rank + 1), N).wait();

    // 7. Synchronous allreduce (blocks until complete on all ranks).
    //    Passing &q as the last argument tells oneCCL to use this SYCL queue
    //    for the Level Zero IPC scale-up operations.
    onecclAllReduce(send, recv, N, onecclBfloat16, onecclSum, comm, &q);
    q.wait();  // ensure GPU work is done before reading result

    if (rank == 0) {
        uint16_t result_elem;
        q.memcpy(&result_elem, recv, sizeof(uint16_t)).wait();
        std::cout << "rank 0 recv[0] (BF16 bits): " << result_elem << "\n";
    }

    // 8. Cleanup.
    sycl::free(send, q);
    sycl::free(recv, q);
    onecclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}
```

Key points about this pattern:

- `onecclGetUniqueId` / `MPI_Bcast` is the standard bootstrap for MPI-launched jobs. For
  jobs without MPI, you would use a shared filesystem or TCP socket to distribute the address.
- `onecclSetDevice(local_rank)` must be called **before** `onecclCommInitRank`. It sets the
  Level Zero device index that oneCCL will use for IPC handle exchange during `topo` scale-up.
- `onecclBfloat16` maps to the `ccl::datatype::bfloat16` selector in the algorithm dispatcher.
- Passing `&q` to `onecclAllReduce` is how you bind the collective to a specific SYCL queue.
  Without it, oneCCL uses an internal default queue.

---

## Overlapping Compute and Communication

For inference, run the collective on one queue while a compute kernel runs on another. This hides allreduce latency behind GEMM time.

The pattern requires **two separate SYCL queues**: one for compute, one for CCL operations.
A SYCL event ties them together:

```cpp
// Two queues: compute kernels run on compute_q, allreduce runs on ccl_q.
// Both are bound to the same physical device.
sycl::queue compute_q(device, {sycl::property::queue::in_order{}});
sycl::queue ccl_q(device,     {sycl::property::queue::in_order{}});

// --- Layer N ---
// Step 1: Submit GEMM (or any compute kernel) to compute_q.
auto gemm_event = compute_q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        send_buf[idx] = /* partial activation output */;
    });
});

// Step 2: Set a cross-queue dependency: ccl_q will not start until gemm_event completes.
//   ext_oneapi_set_external_event inserts a wait on gemm_event at the front of ccl_q.
//   Source: examples/sycl/sycl.cpp lines 129-132
ccl_q.ext_oneapi_set_external_event(gemm_event);

// Step 3: Launch allreduce on ccl_q. This is non-blocking on the HOST —
//   the host thread returns immediately. The allreduce executes on the GPU
//   after gemm_event completes, overlapping with whatever compute_q does next.
onecclAllReduce(send_buf, recv_buf, N, onecclBfloat16, onecclSum, comm, &ccl_q);

// Step 4: Get the allreduce completion event from ccl_q.
auto ar_event_opt = ccl_q.ext_oneapi_get_last_event();
sycl::event ar_event = ar_event_opt.has_value() ? ar_event_opt.value() : sycl::event();

// --- Layer N+1 ---
// Step 5: Submit next compute kernel with explicit dependency on allreduce.
//   This kernel starts only after the allreduce result is in recv_buf.
compute_q.submit([&](sycl::handler& h) {
    h.depends_on(ar_event);
    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        // use recv_buf as input to next layer
    });
});

// Repeat for Layer N+1, N+2, ...
```

The timeline this produces:

```
               Layer N             Layer N+1           Layer N+2
               ─────────────       ─────────────       ─────────────
compute_q:     ████ GEMM N ███     ████ GEMM N+1 ██    ████ GEMM N+2
ccl_q:                  ░░░ AR N ░░░░        ░░░ AR N+1 ░░░░
                         ▲                   ▲
              AR starts after GEMM N done   AR N+1 starts after GEMM N+1 done
              GEMM N+1 starts after AR N done (depends_on)
```

In the ideal overlap case, the GPU stays busy: while AR N runs on `ccl_q`, GEMM N+1 can start on `compute_q` only
if you have work that doesn't depend on the allreduce result (e.g., the QK projection of the
next layer doesn't need the FFN output of this layer). For strict layer-sequential workloads,
the overlap opportunity is smaller — the next layer's GEMM depends on the allreduce output.

**When overlap actually helps:**
- Prefill (large batch): GEMM N+1 is large enough to completely hide AR N
- Double-buffered inference: two requests interleaved, one doing GEMM while the other does AR

**When overlap does not help:**
- Strict sequential layers where every GEMM depends directly on the previous allreduce result
- Decode with batch=1: GEMMs are ~50 µs, AR is ~30 µs — minimal opportunity

---

## Build and Link

```cmake
# CMakeLists.txt fragment
find_package(MPI REQUIRED)
find_package(IntelSYCL REQUIRED)

add_executable(my_engine engine.cpp)

target_compile_options(my_engine PRIVATE -fsycl)
target_link_options(my_engine PRIVATE -fsycl)
target_link_libraries(my_engine PRIVATE
    MPI::MPI_CXX
    oneccl       # links liboneccl.so
    # or: ccl if your install uses that name
)

# Set RPATH so the binary finds liboneccl.so at runtime
set_target_properties(my_engine PROPERTIES
    BUILD_RPATH "$ENV{CCL_ROOT}/lib"
)
```

Or directly with `icpx`:

```bash
source /opt/intel/oneapi/setvars.sh   # sets CCL_ROOT, MPI paths
icpx -fsycl -o my_engine engine.cpp \
     -I$CCL_ROOT/include \
     -L$CCL_ROOT/lib -loneccl \
     $(mpicc --showme:compile) $(mpicc --showme:link)
```

---

## v1 C++ Template API (non-SYCL path)

For CPU-buffer collectives or non-SYCL GPU paths, the v1 C++ API (`ccl::communicator`,
`ccl::allreduce`) is more ergonomic:

```cpp
#include <oneapi/ccl.hpp>   // includes ccl.h with CCL_PRODUCT_FULL defined
#include <mpi.h>

// Init (CPU path, no SYCL queue needed)
MPI_Init(nullptr, nullptr);
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

// Create KVS (key-value store for rank wire-up)
ccl::shared_ptr_class<ccl::kvs> kvs;
ccl::kvs::address_type kvs_addr;
if (rank == 0) {
    kvs = ccl::create_main_kvs();      // creates listening endpoint
    kvs_addr = kvs->get_address();
    MPI_Bcast(kvs_addr.data(), kvs_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
} else {
    MPI_Bcast(kvs_addr.data(), kvs_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    kvs = ccl::create_kvs(kvs_addr);   // connects to rank 0's KVS
}

// Create communicator
auto comm = ccl::create_communicator(size, rank, kvs);

// Allreduce on host buffers (no stream argument = CPU path)
std::vector<float> send(N, static_cast<float>(rank));
std::vector<float> recv(N);
auto event = ccl::allreduce(send.data(), recv.data(), N,
                            ccl::reduction::sum, comm);
event.wait();

MPI_Finalize();
```

The C++ v1 API differs from the v2 C API in:
- `ccl::create_main_kvs()` vs `onecclGetUniqueId` for bootstrap
- `ccl::communicator` object vs `onecclComm_t` handle
- `ccl::event::wait()` vs `q.wait()` for synchronization
- No SYCL queue integration (v2 C API with `&q` is the SYCL path)

> **Source:** `deps/libccl/include/oneapi/ccl/api_functions.hpp` — full signatures for
> `ccl::allreduce`, `ccl::allgather`, `ccl::alltoallv`, `ccl::send`, `ccl::recv`.
> `deps/libccl/include/oneapi/ccl/communicator.hpp` — `ccl::communicator` class.
> `deps/libccl/include/oneapi/ccl/kvs.hpp` — `ccl::kvs` address type and `get_address()`.

---

## Python Wrapper vs Native C++: When to Use Which

| Scenario | Use |
|---|---|
| PyTorch training or inference (vLLM Python path) | `torch.distributed` + CCL backend |
| C++ inference engine with SYCL kernels (the overlapping pattern) | v2 C API (`onecclAllReduce` with `sycl::queue*`) |
| CPU-buffer collective in a C++ tool (benchmarking, preprocessing) | v1 C++ API (`ccl::allreduce`) |
| Debugging / verifying what the Python path calls | Both — `CCL_LOG_LEVEL=debug` is transport-agnostic |

The Python wrapper goes through `oneccl_bindings_for_pytorch` →
`ProcessGroupCCL::allreduce()` → the v1 C++ API → the same algorithm selector and schedule
builder that the native API reaches. The collective hot path is the same C++ schedule execution regardless of entry point —
Python dispatch overhead is not on the critical path once the collective is posted.
