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
            int *read, *flags, *scanout, *write;
            cudaMalloc((void**)&read, n * sizeof(int));
            cudaMalloc((void**)&flags, n * sizeof(int));

            cudaMemcpy(read, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // {
                int s_n = n;
                int og_n = s_n;
                s_n = 1 << ilog2ceil(s_n);
                
                cudaMalloc((void**)&scanout, s_n * sizeof(int));
                int* data = scanout;
                cudaMemset(data, 0, s_n * sizeof(int));
            // }

            timer().startGpuTimer();

            CALCULATE_BLOCK_THREAD_SIZE_AUTO(n, BLOCK_SIZE);
            StreamCompaction::Common::kernMapToBoolean<<<blocksPerGrid, threadsPerBlock>>>(n, flags, read);
            // scan(n, scanout, flags, false);

            // {
                cudaMemcpy(data, flags, og_n * sizeof(int), cudaMemcpyDeviceToDevice);

                int num_iterations = ilog2ceil(s_n);

                // up sweep
                for(int i = 0; i < num_iterations; i++) {
                    int thread_mult = 1 << (i+1);
                    // int thread_mult = 1;

                    CALCULATE_BLOCK_THREAD_SIZE_AUTO((s_n + thread_mult - 1) / thread_mult, BLOCK_SIZE);
                    // printf("KERNEL: %d, %d\n", blocksPerGrid, threadsPerBlock);
                    Reduce<<<blocksPerGrid, threadsPerBlock>>>(s_n, thread_mult, data, 1 << i);
                }

                num_iterations++;

                // down sweep
                cudaMemset(data + (s_n-1), 0, sizeof(int));
                for (int k = 0; k < num_iterations; k++) {
                    int thread_mult = (s_n / (1 << k));
                    // int thread_mult = 1;

                    CALCULATE_BLOCK_THREAD_SIZE((s_n + thread_mult - 1) / (thread_mult), BLOCK_SIZE, blocksPerGrid_new, threadsPerBlock_new);
                    DownSweep<<<blocksPerGrid_new, threadsPerBlock_new>>>(s_n, thread_mult, data, s_n / (1 << k));
                }
            // }
            
            int scanout_end = 0;
            int flags_end = 0;
            cudaMemcpy(&scanout_end, scanout + (n-1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&flags_end, flags + (n-1), sizeof(int), cudaMemcpyDeviceToHost);

            int len = scanout_end + flags_end + 1;
            
            cudaMalloc((void**)&write, len * sizeof(int));
            StreamCompaction::Common::kernScatter<<<blocksPerGrid, threadsPerBlock>>>(n, write, read, flags, scanout);
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            cudaMemcpy(odata, write, len * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(read);
            cudaFree(flags);
            cudaFree(scanout);
            cudaFree(write);

            return len;
        }
    }
}