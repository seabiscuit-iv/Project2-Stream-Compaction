#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/functional.h>
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
            thrust::device_vector<int> dvec_read(n);
            thrust::device_vector<int> dvec_write(n);

            cudaMemcpy(dvec_read.data().get(), idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            thrust::exclusive_scan(dvec_read.begin(), dvec_read.end(), dvec_write.begin());
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            cudaMemcpy(odata, dvec_write.data().get(), n * sizeof(int), cudaMemcpyDeviceToHost);
        }

        struct is_zero {
            __host__ __device__
            bool operator()(int x) const { return x == 0; }
        };

        void compact(int n, int *odata, const int *idata) {
            thrust::device_vector<int> dvec(n);

            cudaMemcpy(dvec.data().get(), idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            auto new_end = thrust::remove_if(dvec.begin(), dvec.end(), is_zero());

            dvec.erase(new_end, dvec.end());
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            cudaMemcpy(odata, dvec.data().get(), dvec.size() * sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
}
