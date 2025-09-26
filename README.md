CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Saahil Gupta
  * [LinkedIn](https://www.linkedin.com/in/saahil-g), [personal website](https://www.saahil-gupta.com)
* Tested on: Windows 11 10.0.26100, AMD Ryzen 9 7940HS @ 4.0GHz 32GB, RTX 4060 Laptop GPU 8GB

## Table of Contents

- [Performance Analysis](#performance-analysis)
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

## Performance Analysis

Performance data was collected for each implementation. CPU timings were measured with `std::chrono`, and GPU timings with `cudaEvents`.

The profiler is implemented in Rust and can be found in `profiling/`. It writes to a cache in `profiling/profile_output/`. *Note that this cache is overwritten on each run and should not be used for multi-configuration results; it only reflects the latest profiling session.*

To run the profiler locally across all configurations, execute `runtests.bat`. This script collects data and generates plot images for each configuration in `img/`.

The graphs below show runtime (ms) against input data size. Data size is plotted on a logscale, where each position corresponds to $2^x$. The y-axis typically ranges from 0–300 ms, but is scaled up for configurations with longer runtimes.

## Basic CPU Implementation

Insert basic implementation description

### Scan

<div align="center">

  ![cpu_scan_256_block_size](img/scan_256_cpu.png)
</div>

### Stream Compaction

<div align="center">

  ![cpu_stream_compaction_256_block_size](img/stream_compaction_256_cpu.png)
</div>


## GPU Naive

Insert GPU Naive implementation description





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

Insert GPU Work Efficient implementation description

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

Insert GPU Work Efficient implementation description

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

Insert GPU Thrust implementation description

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
