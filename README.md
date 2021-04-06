# estimate-pi
Estimate the value of π by monte carlo method.

Some test numbers:
- MacBook Pro (16-inch, 2019), 2.4 GHz 8-Core Intel Core i9, AMD Radeon Pro 5500M 8 GB
- CPU
  - samples = 1,000,000,000
  - 1 thread, 5463.04 ms (1x)
  - 2 threads, 2812.01 ms (1.94x)
  - 4 threads, 1465.39 ms (3.72x)
  - 8 threads, 753.40 ms (7.25x)
  - 16 threads, 553.25 ms (9.87x)
- OpenCL on GPU
  - samples = 1,000,960,000
  - v1 kernel, 258.64 ms (21x)
  - v2 kernel, 32.69 ms (167x)
- Metal
  - samples = 1,000,000,000
  - 35.63 ms (153x)