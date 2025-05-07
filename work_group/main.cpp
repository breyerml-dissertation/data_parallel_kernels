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
        // local memory (explicit)
        sycl::local_accessor<int> loc{ N, cgh };

        cgh.parallel_for(sycl::nd_range<1>{ N, N }, [=](const sycl::nd_item<1> item) {
            // private memory (implicit)
            const int idx = item.get_global_linear_id();
            const int priv = N - idx - 1;

            loc[idx] = d_res[idx];
            // explicit barrier
            sycl::group_barrier(item.get_group());
            d_res[idx] = loc[priv];
        });
    });

    // copy data to the host
    q.copy(d_res, res.data(), N).wait();
    // free the resources
    sycl::free(d_res, q);

    // after swap
    for (int i = 0; i < N; ++i) {
        std::cout << res[i] << ' ';
    }
    std::cout << std::endl;

    return 0;
}
