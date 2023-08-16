#include <stdio.h>
#if defined __ANDROID__ || defined __linux__
#include <sched.h>
#if defined __ANDROID__
#include <dlfcn.h>
#endif
#include <stdint.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <string.h>

#include "mperf/cpu_affinity.h"
#include "mperf/tma/tma.h"

#define FILTER_RADIUS 3

void boxfilter_naive(float* dst, float* src, int width, int height, int radius,
                     bool normalized) {
    const int WIN_SIZE = radius + 1;
    for (int y = 0; y < height; y++) {
        int start_y = y - radius;
        int end_y = y + radius;
        start_y = start_y < 0 ? 0 : start_y;
        end_y = end_y > height - 1 ? height - 1 : end_y;
        for (int x = 0; x < width; x++) {
            int start_x = x - radius;
            int end_x = x + radius;
            start_x = start_x < 0 ? 0 : start_x;
            end_x = end_x > width - 1 ? width - 1 : end_x;
            int sum = 0;
            for (int ky = start_y; ky < end_y; ky++) {
                for (int kx = start_x; kx < end_x; kx++) {
                    sum += src[ky * width + kx];
                }
            }
            if (normalized)
                dst[y * width + x] = sum / (WIN_SIZE * WIN_SIZE);
            else
                dst[y * width + x] = sum;
        }
    }
}

void gettma(int w, int h) {
    printf("----------w:%d, h:%d----------\n", w, h);
    size_t iter_num = 10;

    const size_t buf_size = w * h * sizeof(float);
    float* src = (float*)memalign(4096, buf_size);
    float* dst = (float*)memalign(4096, buf_size);

    for (int i = 0; i < w * h; i++)
        src[i] = (float)(rand() % 10);

    memset(dst, 0, buf_size);
    // warm up
    boxfilter_naive(dst, src, w, h, FILTER_RADIUS, false);

#if defined(__aarch64__)
    mperf::tma::MPFTMA mpf_tma(mperf::MPFXPUType::A55);
    // clang-format off
    mpf_tma.init({"Frontend_Bound",
                      "Fetch_Latency", 
                          "ICache_Misses",
                          "ITLB_Misses",
                          "Predecode_Error",
                      "Fetch_Bandwidth",
                  "Bad_Speculation",
                      "Branch_Mispredicts",
                  "Backend_Bound",
                      "Memory_Bound",
                          "Load_Bound",
                              "Load_DTLB",
                              "Load_Cache",
                          "Store_Bound",
                              "Store_TLB",
                              "Store_Buffer",
                      "Core_Bound",
                          "Interlock_Bound",
                              "Interlock_AGU",
                              "Interlock_FPU",
                          "Core_Bound_Others",
                  "Retiring",
                      "LD_Retiring",
                      "ST_Retiring",
                      "DP_Retiring",
                      "ASE_Retiring",
                      "VFP_Retiring",
                      "PC_Write_Retiring",
                          "BR_IMMED_Retiring",
                          "BR_RETURN_Retiring",
                          "BR_INDIRECT_Retiring",
                "Metric_L1D_Miss_Ratio",	
	                    "Metric_L1D_RD_Miss_Ratio",
	                    "Metric_L1D_WR_Miss_Ratio",
                    "Metric_L2D_Miss_Ratio",	
	                    "Metric_L2D_RD_Miss_Ratio",
	                    "Metric_L2D_WR_Miss_Ratio",
                    "Metric_L3D_Miss_Ratio",	
	                "Metric_L3D_RD_Miss_Ratio",
                    "Metric_BR_Mispred_Ratio",
                    "Metric_L1I_TLB_Miss_Ratio",
                    "Metric_L1D_TLB_Miss_Ratio",
                    "Metric_L2_TLB_Miss_Ratio",
                    "Metric_ITLB_Table_Walk_Ratio",
                    "Metric_DTLB_Table_Walk_Ratio",
                    "Metric_Load_Port_Util",
                    "Metric_Store_Port_Util",
                    "Metric_FPU_Util",
                    "Metric_GFLOPs_Use"});
    // clang-format on
#else
    mperf::tma::MPFTMA mpf_tma(mperf::MPFXPUType::HSX_SERVER);
    mpf_tma.init(
            {"Frontend_Bound", "Bad_Speculation", "Backend_Bound", "Retiring"});
#endif

    size_t gn = mpf_tma.group_num();
    size_t uncore_evt_num = mpf_tma.uncore_events_num();
    for (size_t i = 0; i < gn; ++i) {
        mpf_tma.start(i);
        for (size_t j = 0; j < iter_num; ++j) {
            boxfilter_naive(dst, src, w, h, FILTER_RADIUS, false);
        }
        mpf_tma.sample_and_stop(iter_num);
    }

    for (size_t i = 0; i < uncore_evt_num; ++i) {
        mpf_tma.start_uncore(i);
        for (size_t j = 0; j < iter_num; ++j) {
            boxfilter_naive(dst, src, w, h, FILTER_RADIUS, false);

            mpf_tma.sample(1);
        }
        // mpf_tma.sample_and_stop(iter_num);
    }
    mpf_tma.deinit();

    delete[] src;
    delete[] dst;
}

int main() {
    int dev_id = 0;
    if (set_cpu_thread_affinity_spec_core(dev_id)) {
        printf("faild set thread affinity(core %d)\n", dev_id);
    }
    gettma(1920, 1080);
    gettma(4032, 3024);

    return 0;
}
