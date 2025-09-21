#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        #define DPTR(x) thrust::device_pointer_cast(x)

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            int* d_read;
            int* d_write;

            cudaMalloc((void**)&d_read, n * sizeof(int));
            cudaMalloc((void**)&d_write, n * sizeof(int));

            cudaMemcpy(d_read, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            thrust::exclusive_scan(DPTR(d_read), DPTR(d_read + n), DPTR(d_write));

            timer().endGpuTimer();

            cudaMemcpy(odata, d_write, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(d_read);
            cudaFree(d_write);
        }
    }
}
