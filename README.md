CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Saahil Gupta
  * [LinkedIn](https://www.linkedin.com/in/saahil-g), [personal website](https://www.saahil-gupta.com)
* Tested on: Windows 11 10.0.26100, AMD Ryzen 9 7940HS @ 4.0GHz 32GB, RTX 4060 Laptop GPU 8GB

## Table of Contents

- [Performance Analysis](#performance-analysis)
  - [Basic](#basic)
    - [Scan](#scan)
    - [Stream Compaction](#stream-compaction)
  - [Thread](#thread)
    - [Thread-Efficient Scan](#thread-efficient-scan)
    - [Thread-Efficient Stream Compaction](#thread-efficient-stream-compaction)

## Performance Analysis

### Basic

### Scan

<div align="center">

![scan_256_block_size](img/scan_256.png)
<em>Block Size 256</em>

<table>
  <tr>
    <td>
      <img src="img/scan_128.png" width="400">
      <em>128</em>
    </td>
    <td>
      <img src="img/scan_512.png" width="400">
      <em>512</em>
    </td>
    <td>
      <img src="img/scan_1024.png" width="400">
      <em>1024</em>
    </td>
  </tr>
</table>

</div>

### Stream Compaction

<div align="center">

![compact_256_block_size](img/stream_compaction_256.png)
<em>Block Size 256</em>

<table>
  <tr>
    <td><img src="img/stream_compaction_128.png" width="400">
      <em>128</em>
    </td>
    <td>
      <img src="img/stream_compaction_512.png" width="400">
      <em>512</em>
    </td>
    <td>
      <img src="img/stream_compaction_1024.png" width="400">
      <em>1024</em>
    </td>
  </tr>
</table>

</div>

### Thread Efficiency

### Thread-Efficient Scan


### Thread-Efficient Stream Compaction
