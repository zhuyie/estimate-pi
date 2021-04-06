# estimate-pi
Estimate the value of π by monte carlo method.

Some test numbers:
- MacBook Pro (16-inch, 2019), 2.4 GHz 8-Core Intel Core i9, AMD Radeon Pro 5500M 8 GB
- CPU
  - samples = 1,000,000,000
  - 1 thread, 8820.48 ms (1x)
  - 2 threads, 4457.15 ms (1.97x)
  - 4 threads, 2436.55 ms (3.62x)
  - 8 threads, 1270.31 ms (6.94x)
  - 16 threads, 1044.85 ms (8.44x)
- OpenCL on GPU
  - samples = 1,000,960,000
  - pi_v1 kernel, 258.64 ms (34x)
  - pi_v2 kernel, 32.69 ms (267x)
- Metal
  - samples = 1,000,000,000
  - 35.63 ms (247x)