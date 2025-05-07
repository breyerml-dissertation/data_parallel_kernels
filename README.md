# SYCL Data Parallel Kernels

This repository contains hello world kernels written in SYCL for the different data parallel kernels.

The available data parallel kernels are: 

- **basic** data parallel kernels
- **work-group** data parallel kernels
- **hierarchical** data parallel kernels
- **scoped parallelism** kernels (AdaptiveCpp only)

The supported SYCL implementations are:

- icxp (version 2025.0.0)
- AdaptiveCpp (version v24.10.0)

Each kernel reverses all elements in a vector. 