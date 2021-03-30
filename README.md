# estimate-pi
Estimate the value of π by monte carlo method.

Some test numbers
- MacBook Pro (16-inch, 2019), 2.4 GHz 8-Core Intel Core i9, AMD Radeon Pro 5500M 8 GB
- CPU
  - samples = 1,000,000,000
  - 1 thread, 8820.48ms
  - 2 threads, 4457.15ms
  - 4 threads, 2436.55ms
  - 8 threads, 1270.31ms
  - 16 threads, 1044.85ms
- OpenCL on GPU
  - samples = 1,000,960,000
  - pi_v1 kernel, 258.64ms
  - pi_v2 kernel, 32.69ms
