#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <vector>

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void Reduce(int n, int* data, int k) {
            CALCULATE_TID_AUTO;

            if ( tid >= n || (tid + 1) % (2 * k) != 0 ) {
                return;
            }

            data[tid] = data[tid] + data[tid - k];
        }

        __global__ void DownSweep(int n, int* data, int k) {
            CALCULATE_TID_AUTO;
            tid = 2*tid + 1;

            k = n / (2 << k);

            if ( tid >= n || (tid + 1) % (2 * k) != 0 ) {
                return;
            }

            int temp = data[tid];
            data[tid] = data[tid] + data[tid - k];
            data[tid - k] = temp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool timer_enabled) {
            if (timer_enabled) timer().startGpuTimer();

            int og_n = n;
            n = 1 << ilog2ceil(n);
            
            int* data;
            cudaMalloc((void**)&data, n * sizeof(int));
            cudaMemset(data, 0, n * sizeof(int));
            cudaMemcpy(data, idata, og_n * sizeof(int), cudaMemcpyHostToDevice);

            int num_iterations = ilog2ceil(n);

            CALCULATE_BLOCK_THREAD_SIZE_AUTO(n, BLOCK_SIZE);

            // up sweep
            for(int i = 0; i < num_iterations; i++) {
                Reduce<<<blocksPerGrid, threadsPerBlock>>>(n, data, 1 << i);
            }

            // down sweep
            cudaMemset(data + (n-1), 0, sizeof(int));
            CALCULATE_BLOCK_THREAD_SIZE(n/2, BLOCK_SIZE, blocksPerGrid_new, threadsPerBlock_new);
            for (int k = 0; k < num_iterations; k++) {
                DownSweep<<<blocksPerGrid_new, threadsPerBlock_new>>>(n, data, k);
            }

            cudaMemcpy(odata, data, og_n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(data);

            if (timer_enabled) timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int *read, *flags, *scanout, *write;
            cudaMalloc((void**)&read, n * sizeof(int));
            cudaMalloc((void**)&flags, n * sizeof(int));
            cudaMalloc((void**)&scanout, n * sizeof(int));

            cudaMemcpy(read, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            CALCULATE_BLOCK_THREAD_SIZE_AUTO(n, BLOCK_SIZE);

            StreamCompaction::Common::kernMapToBoolean<<<blocksPerGrid, threadsPerBlock>>>(n, flags, read);

            scan(n, scanout, flags, false);

            int scanout_end = 0;
            int flags_end = 0;
            cudaMemcpy(&scanout_end, scanout + (n-1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&flags_end, flags + (n-1), sizeof(int), cudaMemcpyDeviceToHost);

            int len = scanout_end + flags_end;
            
            cudaMalloc((void**)&write, len * sizeof(int));

            StreamCompaction::Common::kernScatter<<<blocksPerGrid, threadsPerBlock>>>(n, write, read, flags, scanout);

            cudaMemcpy(odata, write, len * sizeof(int), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();

            cudaFree(read);
            cudaFree(flags);
            cudaFree(scanout);
            cudaFree(write);

            timer().endGpuTimer();

            return len;
        }
    }
}
