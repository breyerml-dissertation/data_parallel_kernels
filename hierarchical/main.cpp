#include "sycl/sycl.hpp"

#include <iostream>
#include <vector>

int main() {
    const int N = 24;

    // create and fill used data
    std::vector<double> res(N);
    for (int i = 0; i < N; ++i) {
        res[i] = i;
    }

    // before swap
    for (int i = 0; i < N; ++i) {
        std::cout << res[i] << ' ';
    }
    std::cout << std::endl;

// create a SYCL queue specifying WHERE the code should run
#if defined(EXAMPLE_SYCL_GPU_DEVICE)
    sycl::queue q{ sycl::gpu_selector_v, sycl::property::queue::in_order{} };
#else
    sycl::queue q{ sycl::cpu_selector_v, sycl::property::queue::in_order{} };
#endif
    std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // allocate memory on the device
    double *d_res = sycl::malloc_device<double>(N, q);
    // copy data to the device
    q.copy(res.data(), d_res, N);

    // the SYCL compute kernel
    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for_work_group(sycl::range<1>{ 1 }, sycl::range<1>{ N }, [=](const sycl::group<1> group) {
            // local memory (implicit)
            int loc[N];
            // private memory (explicit)
            sycl::private_memory<int> priv{ group };

            group.parallel_for_work_item([&](sycl::h_item<1> item) {
                // private memory (implicit)
                const int idx = group[0] * group.get_local_range(0) + item.get_local_id(0);
                priv(item) = N - idx - 1;
                loc[idx] = d_res[idx];
            });
            // implicit barrier
            group.parallel_for_work_item([&](sycl::h_item<1> item) {
                // private memory (implicit)
                const int idx = group[0] * group.get_local_range(0) + item.get_local_id(0);
                d_res[idx] = loc[priv(item)];
            });
        });
    });

    // copy data to the host
    q.copy(d_res, res.data(), N);
    // free the resources
    sycl::free(d_res, q);

    // after swap
    for (int i = 0; i < N; ++i) {
        std::cout << res[i] << ' ';
    }
    std::cout << std::endl;

    return 0;
}
