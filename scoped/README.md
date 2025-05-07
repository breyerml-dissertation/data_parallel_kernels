# SYCL

An array reverse example using the AdaptiveCpp scoped parallelism extension.

To use AdaptiveCpp as SYCL compiler, specify `-DEXAMPLE_SYCL_IMPLEMENTATION="acpp"`.

```bash
cmake -DEXAMPLE_SYCL_IMPLEMENTATION="acpp" -DEXAMPLE_SYCL_OFFLOAD_DEVICE_TYPE="gpu" -B build .
cmake --build build
build/hierarchical_reverse
```

The target device, `cpu` or `gpu`, can be switched using the `EXAMPLE_SYCL_OFFLOAD_DEVICE_TYPE` option.

The AdaptiveCpp target must be set using the `ACPP_TARGETS` option:

- JIT compile for the respective target at runtime (recommended): **not supported**
- CPUs: `-DACPP_TARGETS=omp`
- NVIDIA GPUs: `-DACPP_TARGETS=cuda:sm_80`
- AMD GPUs: `-DACPP_TARGETS=hip:gfx90a`
- Intel GPUs: **not supported**
