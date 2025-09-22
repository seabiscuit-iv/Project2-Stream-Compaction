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
            thrust::device_vector<int> dvec_read(n);
            thrust::device_vector<int> dvec_write(n);

            cudaMemcpy(dvec_read.data().get(), idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            thrust::exclusive_scan(dvec_read.begin(), dvec_read.end(), dvec_write.begin());
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            cudaMemcpy(odata, dvec_write.data().get(), n * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
}
