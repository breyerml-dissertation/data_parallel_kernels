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
    double *d_global = sycl::malloc_device<double>(N, q);
    // copy data to the device
    q.copy(res.data(), d_res, N);

    // the SYCL compute kernel
    q.submit([&](sycl::handler &cgh) {
         cgh.parallel_for(sycl::range{ N }, [=](const sycl::item<1> idx) {
             d_global[idx] = d_res[idx];
         });
     }).wait();

    q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::range{ N }, [=](const sycl::item<1> idx) {
            // private memory (implicit)
            const int priv = N - idx - 1;
            d_res[idx] = d_global[priv];
        });
    });

    // copy data to the host
    q.copy(d_res, res.data(), N);
    // free the resources
    sycl::free(d_res, q);
    sycl::free(d_global, q);

    // after swap
    for (int i = 0; i < N; ++i) {
        std::cout << res[i] << ' ';
    }
    std::cout << std::endl;

    return 0;
}
