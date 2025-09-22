#include "thread_efficient.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include <vector>

namespace StreamCompaction {
    namespace ThreadEfficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }


        __global__ void Reduce(int n, int thread_mult, int* data, int k) {
            CALCULATE_TID_AUTO;
            tid = (tid * thread_mult) + (thread_mult - 1);

            if ( tid >= n || tid < k) {
                return;
            }

            data[tid] = data[tid] + data[tid - k];
        }

        __global__ void DownSweep(int n, int thread_mult, int* data, int k) {
            CALCULATE_TID_AUTO;
            tid = (tid * thread_mult) + (thread_mult - 1);

            tid = 2*tid + 1;

            if ( tid >= n || tid < k) {
                return;
            }

            int temp = data[tid];
            data[tid] = data[tid] + data[tid - k];
            data[tid - k] = temp;
        }

        void scan(int n, int *odata, const int *idata, bool timer_enabled) {
            int og_n = n;
            n = 1 << ilog2ceil(n);
            
            int* data;
            cudaMalloc((void**)&data, n * sizeof(int));
            cudaMemset(data, 0, n * sizeof(int));
            cudaMemcpy(data, idata, og_n * sizeof(int), cudaMemcpyHostToDevice);

            if (timer_enabled) timer().startGpuTimer();

            int num_iterations = ilog2ceil(n);

            // up sweep
            for(int i = 0; i < num_iterations; i++) {
                int thread_mult = 1 << (i+1);
                // int thread_mult = 1;

                CALCULATE_BLOCK_THREAD_SIZE_AUTO((n + thread_mult - 1) / thread_mult, BLOCK_SIZE);
                // printf("KERNEL: %d, %d\n", blocksPerGrid, threadsPerBlock);
                Reduce<<<blocksPerGrid, threadsPerBlock>>>(n, thread_mult, data, 1 << i);
            }

            num_iterations++;

            // down sweep
            cudaMemset(data + (n-1), 0, sizeof(int));
            for (int k = 0; k < num_iterations; k++) {
                int thread_mult = (n / (1 << k));
                // int thread_mult = 1;

                CALCULATE_BLOCK_THREAD_SIZE((n + thread_mult - 1) / (thread_mult), BLOCK_SIZE, blocksPerGrid_new, threadsPerBlock_new);
                DownSweep<<<blocksPerGrid_new, threadsPerBlock_new>>>(n, thread_mult, data, n / (1 << k));
            }

            if (timer_enabled) timer().endGpuTimer();

            cudaMemcpy(odata, data, og_n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(data);
        }   

        int compact(int n, int *odata, const int *idata) {
            printf("UNIMPLEMENTED");
            return -1;
        }
    }
}