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
#include <stream_compaction/thread_efficient.h>
#include "testing_helpers.hpp"
#include <vector>
#include <cassert>
#include <fstream>
#include <sstream>

#define FILE_WRITE 1

const int NUM_TESTS = 8;

int SIZE = 1 << 27; // feel free to change the size of array
bool tests[NUM_TESTS] = {true, true, true, true, true, true, true};
int blockSize = 128;

bool testing = false;

void test_gpu_scan_naive() {
    std::vector<int> read(SIZE, 0);
    std::vector<int> h_write(SIZE, 0);
    std::vector<int> d_write(SIZE, 0);

    genArray(SIZE - 1, read.data(), 50);
    read[SIZE - 1] = 0;
    zeroArray(SIZE, h_write.data());
    zeroArray(SIZE, d_write.data());

    StreamCompaction::CPU::scan(SIZE, h_write.data(), read.data());
    // printElapsedTime("CPU Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    StreamCompaction::Naive::scan(SIZE, d_write.data(), read.data());
    // printElapsedTime("GPU Scan Naive: ", StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    assert(h_write == d_write);
    printf("%s passed\n", __func__);
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
    // printElapsedTime("CPU Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    StreamCompaction::Efficient::scan(SIZE, d_write.data(), read.data());
    // printElapsedTime("GPU Scan Work Efficient: ", StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    assert(h_write == d_write);
    printf("%s passed\n", __func__);
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
    // printElapsedTime("CPU Compaction: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    StreamCompaction::Efficient::compact(SIZE, d_write.data(), read.data());
    // printElapsedTime("GPU Compaction Work Efficient: ", StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    assert(h_write == d_write);
    printf("%s passed\n", __func__);
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
    // printElapsedTime("CPU Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    StreamCompaction::Thrust::scan(SIZE, d_write.data(), read.data());
    // printElapsedTime("GPU Thrust Scan: ", StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    assert(h_write == d_write);
    printf("%s passed\n", __func__);
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
    // printElapsedTime("CPU Compaction with Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    StreamCompaction::CPU::compactWithoutScan(SIZE, b_write.data(), read.data());
    // printElapsedTime("CPU Compaction without Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    assert(a_write == b_write);
    printf("%s passed\n", __func__);
}


void test_gpu_stream_compaction_thrust() {
    std::vector<int> read(SIZE, 0);
    std::vector<int> a_write(SIZE, 0);
    std::vector<int> b_write(SIZE, 0);

    genArray(SIZE - 1, read.data(), 50);
    read[SIZE - 1] = 0;
    zeroArray(SIZE, a_write.data());
    zeroArray(SIZE, b_write.data());

    StreamCompaction::CPU::compactWithScan(SIZE, a_write.data(), read.data());
    // printElapsedTime("CPU Compaction with Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    StreamCompaction::Thrust::compact(SIZE, b_write.data(), read.data());
    // printElapsedTime("CPU Compaction without Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    assert(a_write == b_write);
    printf("%s passed\n", __func__);
}

void test_gpu_scan_thread_efficient() {
    std::vector<int> read(SIZE, 0);
    std::vector<int> h_write(SIZE, 0);
    std::vector<int> d_write(SIZE, 0);

    // genArray(SIZE - 1, read.data(), 50);
    onesArray(SIZE - 1, read.data());
    read[SIZE - 1] = 0;
    zeroArray(SIZE, h_write.data());
    zeroArray(SIZE, d_write.data());

    StreamCompaction::CPU::scan(SIZE, h_write.data(), read.data());

    StreamCompaction::ThreadEfficient::scan(SIZE, d_write.data(), read.data());

    assert(h_write == d_write);
    printf("%s passed\n", __func__);
}


void test_gpu_stream_compaction_thread_efficient() {
    std::vector<int> read(SIZE, 0);
    std::vector<int> h_write(SIZE, 0);
    std::vector<int> d_write(SIZE, 0);

    // genArray(SIZE - 1, read.data(), 50);
    onesArray(SIZE - 1, read.data());
    read[SIZE - 1] = 0;
    zeroArray(SIZE, h_write.data());
    zeroArray(SIZE, d_write.data());

    StreamCompaction::CPU::compactWithoutScan(SIZE, h_write.data(), read.data());

    StreamCompaction::ThreadEfficient::compact(SIZE, d_write.data(), read.data());
    assert(h_write == d_write);
    printf("%s passed\n", __func__);
}



void process_command_line_args(int argc, char* argv[]) {
    for ( int arg = 1; arg < argc; arg += 2 ) {
        std::string flag( argv[arg] );

        if ( arg + 1 == argc ) {
            printf("ERROR: No argument provided for flag %s\n", flag);
            std::cout.flush();
        }

        std::string value( argv[arg+1] );

        if ( flag == "-tests" ) {
            testing = true;
            for(int i = 0; i < NUM_TESTS; i++) {
                tests[i] = false;
            }
            if ( value == "CPU_STREAM_COMPACT" ) {
                tests[0] = true;
            } 
            else if ( value == "GPU_SCAN_NAIVE" ) {
                tests[1] = true;
            }
            else if ( value == "GPU_SCAN_EFFICIENT" ) {
                tests[2] = true;
            }
            else if ( value == "GPU_STREAM_COMPACT_EFFICIENT" ) {
                tests[3] = true;
            }
            else if ( value == "GPU_SCAN_THRUST" ) {
                tests[4] = true;
            }
            else if ( value == "GPU_STREAM_COMPACT_THRUST" ) {
                tests[5] = true;
            }
            else if ( value == "GPU_SCAN_THREAD_EFFICIENT" ) {
                tests[6] = true;
            }
            else if ( value == "GPU_STREAM_COMPACT_THREAD_EFFICIENT" ) {
                tests[7] = true;
            }
            else if ( value == "ALL" ) {
                for(int i = 0; i < NUM_TESTS; i++) {
                    tests[i] = true;
                }
            }
            else {
                printf("ERROR: incorrect parameter for flag -tests (CPU_STREAM_COMPACT, GPU_SCAN_NAIVE, GPU_SCAN_EFFICIENT, GPU_STREAM_COMPACT_EFFICIENT, GPU_SCAN_THRUST, GPU_STREAM_COMPACT_THRUST, GPU_SCAN_THREAD_EFFICIENT, GPU_STREAM_COMPACT_THREAD_EFFICIENT, ALL)\n");
                std::cout.flush();
                exit(1);
            }
        }
        else if ( flag == "-size" ) {
            try {
                SIZE = std::stoi(value);
            } catch (...) { 
                printf("ERROR: incorrect value type for flag -size, requires integer\n");
                std::cout.flush();
                exit(1);
            }
        }
        else if ( flag == "-blocksize" ) {
            try {
                blockSize = std::stoi(value);
            } catch (...) { 
                printf("ERROR: incorrect value type for flag -blocksize, requires integer\n");
                std::cout.flush();
                exit(1);
            }
        }
        else {
            printf("ERROR: no known flag %s\n", flag);
            std::cout.flush();
            exit(1);
        }
    }
}






int main(int argc, char* argv[]) {
    for(int i = 0; i < NUM_TESTS; i++) {
        tests[i] = true;
    }

    process_command_line_args(argc, argv);
 
    if (testing) {
        if (tests[0]) test_cpu_stream_compaction();
        if (tests[1]) test_gpu_scan_naive();
        if (tests[2]) test_gpu_scan_work_efficient();
        if (tests[3]) test_gpu_stream_compaction_work_efficient();
        if (tests[4]) test_gpu_scan_thrust();
        if (tests[5]) test_gpu_stream_compaction_thrust();
        if (tests[6]) test_gpu_scan_thread_efficient();
        if (tests[7]) test_gpu_stream_compaction_thread_efficient();
    }
    else { // profiling
        #if FILE_WRITE 
            std::stringstream ss;
            ss << "profiling/profile_output/output_" << blockSize << ".txt";
            std::string path{ss.str()};
            printf("Beginning File Write to %s\n", path.c_str());

            std::ofstream out_file(path);

        for(int i = 5; i <= 27; i++) {
            SIZE = 1 << i;
        #endif

            std::vector<int> read(SIZE, 0);
            std::vector<int> write(SIZE, 0);

            genArray(SIZE - 1, read.data(), 50);
            read[SIZE - 1] = 0;

            const int num_values = 10;
            float values[num_values];

            // CPU Scan
            zeroArray(SIZE, write.data());
            StreamCompaction::CPU::scan(SIZE, write.data(), read.data());
            #if !FILE_WRITE 
                printElapsedTime("CPU Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
            #endif
            values[0] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
        
            // CPU Compaction without Scan
            zeroArray(SIZE, write.data());
            StreamCompaction::CPU::compactWithoutScan(SIZE, write.data(), read.data());
            #if !FILE_WRITE 
                printElapsedTime("CPU Compaction without Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
            #endif
            values[1] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

            // CPU Compaction with Scan
            zeroArray(SIZE, write.data());
            StreamCompaction::CPU::compactWithScan(SIZE, write.data(), read.data());
            #if !FILE_WRITE 
                printElapsedTime("CPU Compaction with Scan: ", StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
            #endif
            values[2] = StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();

            // GPU Scan Naive
            zeroArray(SIZE, write.data());
            StreamCompaction::Naive::scan(SIZE, write.data(), read.data());
            #if !FILE_WRITE 
                printElapsedTime("GPU Scan Naive: ", StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
            #endif
            values[3] = StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();

            // GPU Scan Work Efficient
            zeroArray(SIZE, write.data());
            StreamCompaction::Efficient::scan(SIZE, write.data(), read.data());
            #if !FILE_WRITE 
                printElapsedTime("GPU Scan Work Efficient: ", StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
            #endif
            values[4] = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
        
            // GPU Stream Compaction Work Efficient
            zeroArray(SIZE, write.data());
            StreamCompaction::Efficient::compact(SIZE, write.data(), read.data());
            #if !FILE_WRITE 
                printElapsedTime("GPU Stream Compaction Work Efficient: ", StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
            #endif
            values[5] = StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
        
            // GPU Scan Thread Efficient
            zeroArray(SIZE, write.data());
            StreamCompaction::ThreadEfficient::scan(SIZE, write.data(), read.data());
            #if !FILE_WRITE 
                printElapsedTime("GPU Scan Thread Efficient: ", StreamCompaction::ThreadEfficient::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
            #endif
            values[6] = StreamCompaction::ThreadEfficient::timer().getGpuElapsedTimeForPreviousOperation();

            // GPU Stream Compaction Thread Efficient
            zeroArray(SIZE, write.data());
            StreamCompaction::ThreadEfficient::compact(SIZE, write.data(), read.data());
            #if !FILE_WRITE
                printElapsedTime("GPU Stream Compaction Thread Efficient: ", StreamCompaction::ThreadEfficient::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
            #endif
            values[7] = StreamCompaction::ThreadEfficient::timer().getGpuElapsedTimeForPreviousOperation();

            // GPU Scan Thrust
            zeroArray(SIZE, write.data());
            StreamCompaction::Thrust::scan(SIZE, write.data(), read.data());
            #if !FILE_WRITE 
                printElapsedTime("GPU Scan Thrust: ", StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
            #endif
            values[8] = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();

            // GPU Stream Compaction Thrust
            zeroArray(SIZE, write.data());
            StreamCompaction::Thrust::compact(SIZE, write.data(), read.data());
            #if !FILE_WRITE 
                printElapsedTime("GPU Stream Compaction Thrust: ", StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
            #endif
            values[9] = StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();

        #if FILE_WRITE
            for (int i = 0; i < num_values; i++) {
                out_file << values[i] << " ";
            }

            out_file << i << std::endl;
        }

        out_file.close();
        

        #if FILE_WRITE
            printf("Finished File Write to %s\n", path.c_str());
        #endif

        #endif
    }
}
