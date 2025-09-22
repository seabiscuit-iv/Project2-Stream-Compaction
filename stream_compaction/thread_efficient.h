#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace ThreadEfficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata, bool timer_enabled = true);

        int compact(int n, int *odata, const int *idata);
    }
}