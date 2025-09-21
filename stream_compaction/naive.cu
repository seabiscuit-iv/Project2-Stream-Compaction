#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <cmath>

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__


        __global__ void ScanIteration(int n, int* read, int* write, int k) {
            CALCULATE_TID_AUTO;

            if ( tid >= n ) {
                return;
            }

            if ( tid < k ) {
                write[tid] = read[tid];
                return;
            }

            write[tid] = read[tid] + read[tid - k];
        }

        __global__ void InclusiveToExclusiveShift(int n, int* read, int* write) {
            CALCULATE_TID_AUTO;

            if ( tid >= n ) {
                return;
            }

            if ( tid == 0 ) {
                write[tid] = 0;
            } 

            write[tid] = read[tid - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int* d_a;
            int* d_b;

            cudaMalloc((void**)&d_a, n * sizeof(int));
            cudaMalloc((void**)&d_b, n * sizeof(int));

            checkCUDAError("Naive::Scan cudaMalloc failed");

            cudaMemcpy(d_a, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            checkCUDAError("Naive::Scan cudaMemcpy failed");

            int num_iterations = ilog2ceil(n);

            CALCULATE_BLOCK_THREAD_SIZE_AUTO(n, BLOCK_SIZE);

            for (int i = 0; i < num_iterations; i++) {
                int* d_read = i % 2 == 0 ? d_a : d_b;
                int* d_write = i % 2 == 0 ? d_b : d_a;

                int k = 1 << i;

                ScanIteration<<<blocksPerGrid, threadsPerBlock>>>(n, d_read, d_write, k);
                checkCUDAError("Naive::Scan ScanIteration failed");
            }
            int* d_read = num_iterations % 2 == 0 ? d_a : d_b;
            int* d_write = num_iterations % 2 == 0 ? d_b : d_a;

            InclusiveToExclusiveShift<<<blocksPerGrid, threadsPerBlock>>>(n, d_read, d_write);

            cudaMemcpy(odata, d_write, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Naive::Scan cudaMemcpy failed");

            cudaDeviceSynchronize();

            cudaFree(d_a);
            cudaFree(d_b);

            timer().endGpuTimer();
        }
    }
}
