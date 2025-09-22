CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Saahil Gupta
  * [LinkedIn](https://www.linkedin.com/in/saahil-g), [personal website](https://www.saahil-gupta.com)
* Tested on: Windows 11 10.0.26100, AMD Ryzen 9 7940HS @ 4.0GHz 32GB, RTX 4060 Laptop GPU 8GB

## Performance Analysis

### Scan

---

<div align="center">

![scan_256_block_size](img/scan_256.png)

<table>
  <tr>
    <td><img src="img/scan_128.png" width="400"></td>
    <td><img src="img/scan_512.png" width="400"></td>
    <td><img src="img/scan_1024.png" width="400"></td>
  </tr>
</table>

</div>

### Stream Compaction

---

<div align="center">

![compact_256_block_size](img/compact_256.png)

<table>
  <tr>
    <td><img src="img/compact_128.png" width="400"></td>
    <td><img src="img/compact_512.png" width="400"></td>
    <td><img src="img/compact_1024.png" width="400"></td>
  </tr>
</table>

</div>