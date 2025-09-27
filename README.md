CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Saahil Gupta
  * [LinkedIn](https://www.linkedin.com/in/saahil-g), [personal website](https://www.saahil-gupta.com)
* Tested on: Windows 11 10.0.26100, AMD Ryzen 9 7940HS @ 4.0GHz 32GB, RTX 4060 Laptop GPU 8GB

## Table of Contents

- [Overview](#overview)
- [Implementation and Performance Analysis](#implementation-and-performance-analysis)
  - [CPU](#basic-cpu-implementation)
    - [Scan](#scan)
    - [Stream Compaction](#stream-compaction)
  - [GPU Naive](#gpu-naive)
    - [Scan](#scan-1)
  - [GPU Work Efficient](#gpu-work-efficiency)
    - [Scan](#scan-2)
    - [Stream Compaction](#stream-compaction-1)
  - [GPU Thread Efficient](#gpu-thread-efficient)  
    - [Scan](#scan-3)
    - [Stream Compaction](#stream-compaction-2)
  - [GPU Thrust](#gpu-thrust)
    - [Scan](#scan-4)
    - [Stream Compaction](#stream-compaction-3)
- [Potential Future Optimizations](#potential-future-optimizations)
  - [Global Memory Coalescing](#global-memory-coalescing)
  - [Shared Memory](#shared-memory)
  - [Block Based Division](#block-based-division)



## Overview
This project implements and benchmarks multiple algorithms for **stream compaction** and **scan** on both the CPU and GPU. Stream compaction is the process of removing unwanted elements from an array(*in our case, removing zeroes from an array of integers*) while maintaining the order of remaining elements, and scan computes prefix sums of an array.

This project contains the following versions of `scan` and `compact`:

- **CPU**: iterative scan and stream compaction using standard sequential methods
- **GPU Naive**: a straightforward parallel implementation using simple GPU scan kernels  
- **GPU Work-Efficient**: an optimized parallel algorithm that reduces redundant work
- **GPU Thread-Efficient**: further optimizes GPU execution by minimizing idle threads through striding
- **GPU Thrust**: leverages NVIDIA’s Thrust library for highly optimized scan and compaction routines


## Implementation and Performance Analysis

Performance data was collected for each implementation. CPU timings were measured with `std::chrono`, and GPU timings with `cudaEvents`.

The profiler is implemented in Rust and can be found in `profiling/`. It writes to a cache in `profiling/profile_output/`. *Note that this cache is overwritten on each run and should not be used for multi-configuration results; it only reflects the latest profiling session.*

To run the profiler locally across all configurations, execute `runtests.bat`. This script collects data and generates plot images for each configuration in `img/`.

The graphs below show runtime (ms) against input data size. Data size is plotted on a logscale, where each position corresponds to $2^x$. The y-axis typically ranges from 0–300 ms, but is scaled up for configurations with longer runtimes.

## Basic CPU Implementation

The CPU implementations can be found in `src/cpu.cu`. 

1. `StreamCompaction::CPU::scan`  
Implements a standard prefix-sum (scan) algorithm, starting at the beginning of the array and accumulating values iteratively:
```py
scan(data) -> out:
  out[0] = 0;
  for i in 1..n do
    out[i] = out[i-1] + data[i-1]
```

2. `StreamCompaction::CPU::compactWithoutScan`  
Performs stream compaction without using scan, by iteratively appending nonzero values to an output array:
```py
compactWithoutScan(data) -> out, len:
  c = 0
  for i in 0..n do
    if data[i] != 0 then
      out[c] = data[i]
      c += 1
  
  len = c
```

3. `StreamCompaction::CPU::compactWithScan`  
Implements stream compaction using the scan function, resembling the parallel algorithm used for the GPU implementation:
```py
compactWithScan(data) -> out, len:
  flags = [0; n]
  scanout = [0; n]

  for i in 0..n do
    flags[i] = if data[i] == 0 then 0 else 1

  scanout = scan(flags)

  for i in 0..n do
    if flags[i] == 1 then
      out[scanout[i]] = data[i]

  len = scanout[n-1] + flags[n-1]
```

### Scan

<div align="center">

  ![cpu_scan_256_block_size](img/scan_256_cpu.png)
</div>

### Stream Compaction

<div align="center">

  ![cpu_stream_compaction_256_block_size](img/stream_compaction_256_cpu.png)
</div>


## GPU Naive

The GPU Naive implementations can be found in `src/naive.cu`.

1. `StreamCompaction::Naive::scan`  
Implements scan by precomputing sum for a window size `k` for every index `i > k`, where `k` doubles every iteration.

![gpu_naive_img](img/figure-39-2.jpg)

*Naive stream compaction was omitted, since the only difference would be a call to the naive scan function (naive vs work efficient).*

### Scan

<div align="center">

![naive_scan_256_block_size](img/scan_256_naive.png)
<em>Block Size 256</em>

<table>
  <tr>
    <td>
      <img src="img/scan_128_naive.png" width="400">
      <em>128</em>
    </td>
    <td>
      <img src="img/scan_512_naive.png" width="400">
      <em>512</em>
    </td>
    <td>
      <img src="img/scan_1024_naive.png" width="400">
      <em>1024</em>
    </td>
  </tr>
</table>

</div>




## GPU Work Efficiency

The GPU Work Efficient implementations can be found in `src/efficient.cu`.

1. `StreamCompaction::Efficient::scan`  
Implements scan using two phases: an **up-sweep** (reduction/sum) followed by a **down-sweep** (reconstruction of the prefix sum array). For a detailed explanation of this algorithm, see [GPU Gems 3, Chapter 39](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html).

2. `StreamCompaction::Efficient::compact`
Implements stream compaction by parallelizing the CPU algorithm above, converting the map and scatter parts into kernels, and making use of work-efficient scan.

<div align="center">

![work_efficient_scan_256_block_size](img/scan_256_work_efficient.png)
<em>Block Size 256</em>

<table>
  <tr>
    <td>
      <img src="img/scan_128_work_efficient.png" width="400">
      <em>128</em>
    </td>
    <td>
      <img src="img/scan_512_work_efficient.png" width="400">
      <em>512</em>
    </td>
    <td>
      <img src="img/scan_1024_work_efficient.png" width="400">
      <em>1024</em>
    </td>
  </tr>
</table>

</div>

### Stream Compaction

<div align="center">

![work_efficient_compact_256_block_size](img/stream_compaction_256_work_efficient.png)
<em>Block Size 256</em>

<table>
  <tr>
    <td><img src="img/stream_compaction_128_work_efficient.png" width="400">
      <em>128</em>
    </td>
    <td>
      <img src="img/stream_compaction_512_work_efficient.png" width="400">
      <em>512</em>
    </td>
    <td>
      <img src="img/stream_compaction_1024_work_efficient.png" width="400">
      <em>1024</em>
    </td>
  </tr>
</table>

</div>










## GPU Thread Efficient

The GPU Thread Efficient implementations can be found in `src/thread_efficient.cu`.

These implementations use the same algorithms as `StreamCompaction::Efficient`, but apply **striding** to minimize idle threads.  

- In the **up-sweep**, the stride doubles each iteration.  
- In the **down-sweep**, the stride halves each iteration.  
- The thread index for a given stride is calculated as `tid = (tid * stride) + (stride - 1);`

### Scan

<div align="center">

![thread_efficient_scan_256_block_size](img/scan_256_thread_efficient.png)
<em>Block Size 256</em>

<table>
  <tr>
    <td>
      <img src="img/scan_128_thread_efficient.png" width="400">
      <em>128</em>
    </td>
    <td>
      <img src="img/scan_512_thread_efficient.png" width="400">
      <em>512</em>
    </td>
    <td>
      <img src="img/scan_1024_thread_efficient.png" width="400">
      <em>1024</em>
    </td>
  </tr>
</table>

</div>

### Stream Compaction

<div align="center">

![thread_efficient_compact_256_block_size](img/stream_compaction_256_thread_efficient.png)
<em>Block Size 256</em>

<table>
  <tr>
    <td><img src="img/stream_compaction_128_thread_efficient.png" width="400">
      <em>128</em>
    </td>
    <td>
      <img src="img/stream_compaction_512_thread_efficient.png" width="400">
      <em>512</em>
    </td>
    <td>
      <img src="img/stream_compaction_1024_thread_efficient.png" width="400">
      <em>1024</em>
    </td>
  </tr>
</table>

</div>






## GPU Thrust

The GPU Thrust implementations can be found in `src/thrust.cu`.

1. `StreamCompaction::Thrust::scan`  
Implements scan using `thrust::exclusive_scan`

2. `StreamCompaction::Thrust::compact`
Implements stream compaction using `thrust::remove_if`

Both of these functions have significantly improved speed by using `thrust::device_vector` instead of manual calls to `cudaMalloc` and `cudaFree`

<div align="center">

![thrust_scan_256_block_size](img/scan_256_thrust.png)
<em>Block Size 256</em>

<table>
  <tr>
    <td>
      <img src="img/scan_128_thrust.png" width="400">
      <em>128</em>
    </td>
    <td>
      <img src="img/scan_512_thrust.png" width="400">
      <em>512</em>
    </td>
    <td>
      <img src="img/scan_1024_thrust.png" width="400">
      <em>1024</em>
    </td>
  </tr>
</table>

</div>

### Stream Compaction

<div align="center">

![thrust_compact_256_block_size](img/stream_compaction_256_thrust.png)
<em>Block Size 256</em>

<table>
  <tr>
    <td><img src="img/stream_compaction_128_thrust.png" width="400">
      <em>128</em>
    </td>
    <td>
      <img src="img/stream_compaction_512_thrust.png" width="400">
      <em>512</em>
    </td>
    <td>
      <img src="img/stream_compaction_1024_thrust.png" width="400">
      <em>1024</em>
    </td>
  </tr>
</table>

</div>


## Potential Future Optimizations

The thrust implementation for `scan` and `compact` is highly optimized and well maintained by Nvidia, and considered to be the gold standard for runtime on algorithms. The data collected shows a lot of room for further optimization and reducing runtime to ~10ms, even on extremely large input data sizes. There are plenty of future optimizations we can add to keep up with thrust.

### Global Memory Coalescing

In the current thread-efficient `scan`, striding reduces idle threads but increases the spacing between reads and writes, where now for a stride of `k`, each read/write is spaced out by `k` cells. This results in poor cache utilization and uncoalesced global memory accesses.

For the up-sweep portion, values cannot be discarded. But since we can easily precompute the number of times an index will be acted upon, a possible improvement is to remap our indices so that they sort in descending order of number of kernel operations applied to it. This would result in every kernel having one-to-one memory coalescing for writes. For reads, while not perfect, would still be near and far more cache friendly than an unoptimized version. Down-sweep could likely benefit from a similar remapping. The array could be reorganized using a single remap kernel call. 

This optimization is straightforward for pow-2 array sizes, but less so for arbitrary array sizes. Currently we pad our data array to the next power of two, but this could break down if combined with the [block based division optimization](#block-based-division)


### Shared Memory

The thrust implementation for both `scan` and `compact` both use a heavy amount of shared memory as a block-local cache for global memory. Implementing this would likely show a noticeable change across increasing block size. It's also possible within this to reduce the number of bank conflicts by changing the memory access pattern. More details on the exact implementation can be found [here](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html).


### Block Based Division

Our current `scan` algorithm requires array lengths to be powers of two, forcing us to pad our input to the next power of two and potentially double memory usage. A more scalable approach would be block-based division, where we allocate blocks to work on pow-2 sections of the array, and then run a second-pass kernel that will work on its finished outputs. The important distinction here is that the number of blocks we can spawn is free to be any integer, so at most the number of threads and memory bytes we waste is `BLOCK_SIZE - 1`.