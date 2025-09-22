/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"
#include <vector>
#include <cassert>

const int SIZE = 1 << 25; // feel free to change the size of array

void test_gpu_scan_naive() {
    std::vector<int> read(SIZE, 0);
    std::vector<int> h_write(SIZE, 0);
    std::vector<int> d_write(SIZE, 0);

    genArray(SIZE - 1, read.data(), 50);
    read[SIZE - 1] = 0;
    zeroArray(SIZE, h_write.data());
    zeroArray(SIZE, d_write.data());

    StreamCompaction::CPU::scan(SIZE, h_write.data(), read.data());
    printElapsedTime("CPU Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    StreamCompaction::Naive::scan(SIZE, d_write.data(), read.data());
    printElapsedTime("GPU Scan Naive: ", StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    assert(h_write == d_write);
}

void test_gpu_scan_work_efficient() {
    std::vector<int> read(SIZE, 0);
    std::vector<int> h_write(SIZE, 0);
    std::vector<int> d_write(SIZE, 0);

    genArray(SIZE - 1, read.data(), 50);
    read[SIZE - 1] = 0;
    zeroArray(SIZE, h_write.data());
    zeroArray(SIZE, d_write.data());

    StreamCompaction::CPU::scan(SIZE, h_write.data(), read.data());
    printElapsedTime("CPU Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    StreamCompaction::Efficient::scan(SIZE, d_write.data(), read.data());
    printElapsedTime("GPU Scan Work Efficient: ", StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    assert(h_write == d_write);
}


void test_gpu_stream_compaction_work_efficient() {
    std::vector<int> read(SIZE, 0);
    std::vector<int> h_write(SIZE, 0);
    std::vector<int> d_write(SIZE, 0);

    genArray(SIZE - 1, read.data(), 50);
    read[SIZE - 1] = 0;
    zeroArray(SIZE, h_write.data());
    zeroArray(SIZE, d_write.data());

    StreamCompaction::CPU::compactWithScan(SIZE, h_write.data(), read.data());
    printElapsedTime("CPU Compaction: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    StreamCompaction::Efficient::compact(SIZE, d_write.data(), read.data());
    printElapsedTime("GPU Compaction Work Efficient: ", StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    assert(h_write == d_write);
}


void test_gpu_scan_thrust() {
    std::vector<int> read(SIZE, 0);
    std::vector<int> h_write(SIZE, 0);
    std::vector<int> d_write(SIZE, 0);

    genArray(SIZE - 1, read.data(), 50);
    read[SIZE - 1] = 0;
    zeroArray(SIZE, h_write.data());
    zeroArray(SIZE, d_write.data());

    StreamCompaction::CPU::scan(SIZE, h_write.data(), read.data());
    printElapsedTime("CPU Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    StreamCompaction::Thrust::scan(SIZE, d_write.data(), read.data());
    printElapsedTime("GPU Thrust Scan: ", StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    assert(h_write == d_write);
}

void test_cpu_stream_compaction() {
    std::vector<int> read(SIZE, 0);
    std::vector<int> a_write(SIZE, 0);
    std::vector<int> b_write(SIZE, 0);

    genArray(SIZE - 1, read.data(), 50);
    read[SIZE - 1] = 0;
    zeroArray(SIZE, a_write.data());
    zeroArray(SIZE, b_write.data());

    StreamCompaction::CPU::compactWithScan(SIZE, a_write.data(), read.data());
    printElapsedTime("CPU Compaction with Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    StreamCompaction::CPU::compactWithoutScan(SIZE, b_write.data(), read.data());
    printElapsedTime("CPU Compaction without Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    assert(a_write == b_write);
}



const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

int main(int argc, char* argv[]) {
 
    test_cpu_stream_compaction();
    test_gpu_scan_naive();
    test_gpu_scan_work_efficient();
    test_gpu_stream_compaction_work_efficient();
    test_gpu_scan_thrust();

    system("pause");
}
