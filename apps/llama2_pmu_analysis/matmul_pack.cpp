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
#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>

#include <arm_neon.h>
#include "mperf/cpu_affinity.h"
#include "mperf/exception.h"
#include "mperf/timer.h"
#include "mperf/tma/tma.h"

#define W(i, j) w[(i)*ldw + (j)]
#define X(i) x[(i)]
#define Y(i) xout[(i)]

#define kc 64  // fixme: kc=L1D/(mr+nr) -> kc=L1D/mr=32*1024B/8*4*4B=256

static int unroll_num = 8;
static int vec_size = 4;

void matmul_naive(float* xout, float* x, float* w, int ldw, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += W(i, j) * X(j);
        }
        Y(i) = val;
    }
}

float check(float* xout, float* x, float* w, int n, int d) {
    const size_t buf_size = d * sizeof(float);
    float* ans = (float*)memalign(4096, buf_size);
    memset(ans, 0, buf_size);
    matmul_naive(ans, x, w, n, n, d);
    float max_err = 1e-6;
    float err;
    for (int i = 0; i < d; i++) {
        err = std::abs(xout[i] - ans[i]);
        if (err > max_err) {
            max_err = err;
            // printf("%d: %f vs %f \n", i, xout[i], ans[i]);
        }
    }
    delete[] ans;
    return max_err;
}

/* Routine for computing Y = W * x */

void AddDot8x4_packed(int, float*, float*, float*, int);
void AddDot8x8(int, float*, float*, float*, int);
void AddDot4x8(int, float*, float*, float*, int);
void AddDot4x16(int, float*, float*, float*, int);
void matmul_unroll_packed(float*, float*, float*, int, int, int);

void matmul_pack(float* xout, float* x, float* w, int ldw, int n, int d) {
    int pb = 0;
    for (int p = 0; p < n; p += kc) {
        pb = std::min(kc, n - p);
        matmul_unroll_packed(&Y(0), &X(p), &W(0, p), ldw, pb, d);
    }
}

void gettma(int n, int d) {
    printf("----------d:%d, n:%d----------\n", d, n);
    size_t iter_num = 10;

    const size_t buf_size_w = d * n * sizeof(float);
    const size_t buf_size_x = n * 1 * sizeof(float);
    const size_t buf_size_y = d * 1 * sizeof(float);
    float* w = (float*)memalign(4096, buf_size_w);
    float* x = (float*)memalign(4096, buf_size_x);
    float* y = (float*)memalign(4096, buf_size_y);

    for (int i = 0; i < d * n; i++)
        w[i] = (float)(rand() % 10);
    for (int i = 0; i < n * 1; i++)
        x[i] = (float)(rand() % 10);
    memset(y, 0, d * 1 * sizeof(float));
    // warm up
    matmul_block(y, x, w, n, n, d);

    float max_err = check(y, x, w, n, d);
    printf("max_err:%f\n", max_err);
    if (max_err > 0.1f) {
        mperf_throw(mperf::MperfError, "ERROR: result check error.");
    }
    // must clean up it
    memset(y, 0, d * 1 * sizeof(float));
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
            matmul_block(y, x, w, n, n, d);
        }
        mpf_tma.sample_and_stop(iter_num);
    }

    for (size_t i = 0; i < uncore_evt_num; ++i) {
        mpf_tma.start_uncore(i);
        for (size_t j = 0; j < iter_num; ++j) {
            matmul_block(y, x, w, n, n, d);

            mpf_tma.sample(1);
        }
        // mpf_tma.sample_and_stop(iter_num);
    }
    mpf_tma.deinit();

    delete[] w;
    delete[] x;
    delete[] y;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr,
                "./llama2_cpu_matmul_block core_id unroll_rows vec_size\n");
        return -1;
    }
    const int dev_id = atoi(argv[1]);
    unroll_num = atoi(argv[2]);
    vec_size = atoi(argv[3]);

    printf("unroll: %d lines \nvectorize: %d floats\n", unroll_num, vec_size);

    if (set_cpu_thread_affinity_spec_core(dev_id)) {
        printf("faild set thread affinity(core %d)\n", dev_id);
    }

    gettma(288, 288);
    gettma(512, 512);
    gettma(768, 768);

    return 0;
}

void matmul_unroll_packed(float* xout, float* x, float* w, int ldw, int n, int d) {
    int i = 0;
    // 循环展开，一次处理的行数
    if (unroll_num == 8) {
        for (i = 0; i < d; i += 8) {
            if (i + 8 > d)
                break;
            if (vec_size == 4)
                AddDot8x4_packed(n, &Y(i), &X(0), &W(i, 0), ldw);
            else if (vec_size == 8)
                AddDot8x8(n, &Y(i), &X(0), &W(i, 0), ldw);
            else
                fprintf(stderr, "unsupported AddDot8x%d\n", vec_size);
        }
    } else if (unroll_num == 4) {
        for (i = 0; i < d; i += 4) {
            if (i + 4 > d)
                break;
            if (vec_size == 8)
                AddDot4x8(n, &Y(i), &X(0), &W(i, 0), ldw);
            else if (vec_size == 16)
                AddDot4x16(n, &Y(i), &X(0), &W(i, 0), ldw);
            else
                fprintf(stderr, "unsupported AddDot4x%d\n", vec_size);
        }
    } else {
        fprintf(stderr, "unsupported unroll %d lines\n", unroll_num);
    }
    if (i != d) {
        printf("%s:%d %d != %d\n", __FILE__, __LINE__, i, d);
    }
}

void AddDot8x4_packed(int n, float* xout, float* x, float* w, int ldw) {
    int i;

    // first, pack this block data
    float* wp = (float*)memalign(4096, 8*n*sizeof(float));
    float* w_nptr_l0 = w + ldw * 0;
    float* w_nptr_l1 = w + ldw * 1;
    float* w_nptr_l2 = w + ldw * 2;
    float* w_nptr_l3 = w + ldw * 3;
    float* w_nptr_l4 = w + ldw * 4;
    float* w_nptr_l5 = w + ldw * 5;
    float* w_nptr_l6 = w + ldw * 6;
    float* w_nptr_l7 = w + ldw * 7;

    for (int i = 0; i < n; i += 4) {
        if (i + 4 > n)
            break;
        memcpy(wp + 4 * 0, w_nptr_l0, 4 * sizeof(float));
        memcpy(wp + 4 * 1, w_nptr_l1, 4 * sizeof(float));
        memcpy(wp + 4 * 2, w_nptr_l2, 4 * sizeof(float));
        memcpy(wp + 4 * 3, w_nptr_l3, 4 * sizeof(float));
        memcpy(wp + 4 * 4, w_nptr_l4, 4 * sizeof(float));
        memcpy(wp + 4 * 5, w_nptr_l5, 4 * sizeof(float));
        memcpy(wp + 4 * 6, w_nptr_l6, 4 * sizeof(float));
        memcpy(wp + 4 * 7, w_nptr_l7, 4 * sizeof(float));

        w_nptr_l0 += 4;
        w_nptr_l1 += 4;
        w_nptr_l2 += 4;
        w_nptr_l3 += 4;
        w_nptr_l4 += 4;
        w_nptr_l5 += 4;
        w_nptr_l6 += 4;
        w_nptr_l7 += 4;
    }

    // 先暂存循环展开的每行数据，整行处理完得到最终结果再存到内存。
    // 注：也可以每次写入xout，下一个循环再读出来累加，但效率不高
    float32x4_t y_l00_sum = {0};
    float32x4_t y_l10_sum = {0};
    float32x4_t y_l20_sum = {0};
    float32x4_t y_l30_sum = {0};
    float32x4_t y_l40_sum = {0};
    float32x4_t y_l50_sum = {0};
    float32x4_t y_l60_sum = {0};
    float32x4_t y_l70_sum = {0};

    for (i = 0; i < n; i += 4) {
        if (i + 4 > n)
            break;

        // 加载 X 数据
        float32x4_t x_reg0 = vld1q_f32(&X(i));

        // 每行起始地址，按循环指定的stride跳跃
        float32x4_t w_l00_reg0 = vld1q_f32(&W(0, i));
        y_l00_sum = vfmaq_f32(y_l00_sum, w_l00_reg0, x_reg0);
        float32x4_t w_l10_reg0 = vld1q_f32(&W(1, i));
        y_l10_sum = vfmaq_f32(y_l10_sum, w_l10_reg0, x_reg0);
        float32x4_t w_l20_reg0 = vld1q_f32(&W(2, i));
        y_l20_sum = vfmaq_f32(y_l20_sum, w_l20_reg0, x_reg0);
        float32x4_t w_l30_reg0 = vld1q_f32(&W(3, i));
        y_l30_sum = vfmaq_f32(y_l30_sum, w_l30_reg0, x_reg0);
        float32x4_t w_l40_reg0 = vld1q_f32(&W(4, i));
        y_l40_sum = vfmaq_f32(y_l40_sum, w_l40_reg0, x_reg0);
        float32x4_t w_l50_reg0 = vld1q_f32(&W(5, i));
        y_l50_sum = vfmaq_f32(y_l50_sum, w_l50_reg0, x_reg0);
        float32x4_t w_l60_reg0 = vld1q_f32(&W(6, i));
        y_l60_sum = vfmaq_f32(y_l60_sum, w_l60_reg0, x_reg0);
        float32x4_t w_l70_reg0 = vld1q_f32(&W(7, i));
        y_l70_sum = vfmaq_f32(y_l70_sum, w_l70_reg0, x_reg0);
    }
    if (i != n) {
        printf("%s:%d %d != %d\n", __FILE__, __LINE__, i, n);
    }

    // float* y_ptr = xout;
    // *y_ptr++ = vaddvq_f32(y_l00_sum);
    // *y_ptr++ = vaddvq_f32(y_l10_sum);
    // *y_ptr++ = vaddvq_f32(y_l20_sum);
    // *y_ptr++ = vaddvq_f32(y_l30_sum);
    // *y_ptr++ = vaddvq_f32(y_l40_sum);
    // *y_ptr++ = vaddvq_f32(y_l50_sum);
    // *y_ptr++ = vaddvq_f32(y_l60_sum);
    // *y_ptr++ = vaddvq_f32(y_l70_sum);

    float* y_ptr = xout;
    float32x4_t y_reg0 = vld1q_f32(y_ptr);
    float32x4_t y_reg4 = vld1q_f32(y_ptr + 4);
    float32x4_t zeros = vdupq_n_f32(0.0f);
    float32x4_t ones = vdupq_n_f32(1.0f);

    float32_t y_sum = vaddvq_f32(y_l00_sum);
    float32x4_t mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 0);
    float32x4_t y_res0 = vfmaq_n_f32(y_reg0, mask, y_sum);

    y_sum = vaddvq_f32(y_l10_sum);
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 1);
    y_res0 = vfmaq_n_f32(y_res0, mask, y_sum);
    y_sum = vaddvq_f32(y_l20_sum);
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 2);
    y_res0 = vfmaq_n_f32(y_res0, mask, y_sum);
    y_sum = vaddvq_f32(y_l30_sum);
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 3);
    y_res0 = vfmaq_n_f32(y_res0, mask, y_sum);
    vst1q_f32(y_ptr, y_res0);

    y_sum = vaddvq_f32(y_l40_sum);
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 0);
    float32x4_t y_res1 = vfmaq_n_f32(y_reg4, mask, y_sum);
    y_sum = vaddvq_f32(y_l50_sum);
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 1);
    y_res1 = vfmaq_n_f32(y_res1, mask, y_sum);
    y_sum = vaddvq_f32(y_l60_sum);
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 2);
    y_res1 = vfmaq_n_f32(y_res1, mask, y_sum);
    y_sum = vaddvq_f32(y_l70_sum);
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 3);
    y_res1 = vfmaq_n_f32(y_res1, mask, y_sum);
    vst1q_f32(y_ptr + 4, y_res1);
}

void AddDot8x8(int n, float* xout, float* x, float* w, int ldw) {
    int i;

    // 先暂存循环展开的每行数据，整行处理完得到最终结果再存到内存。
    // 注：也可以每次写入xout，下一个循环再读出来累加，但效率不高
    float32x4_t y_l00_sum = {0};
    float32x4_t y_l04_sum = {0};
    float32x4_t y_l10_sum = {0};
    float32x4_t y_l14_sum = {0};
    float32x4_t y_l20_sum = {0};
    float32x4_t y_l24_sum = {0};
    float32x4_t y_l30_sum = {0};
    float32x4_t y_l34_sum = {0};
    float32x4_t y_l40_sum = {0};
    float32x4_t y_l44_sum = {0};
    float32x4_t y_l50_sum = {0};
    float32x4_t y_l54_sum = {0};
    float32x4_t y_l60_sum = {0};
    float32x4_t y_l64_sum = {0};
    float32x4_t y_l70_sum = {0};
    float32x4_t y_l74_sum = {0};

    for (i = 0; i < n; i += 8) {
        if (i + 8 > n)
            break;

        // 加载 X 数据
        float32x4_t x_reg0 = vld1q_f32(&X(i));
        float32x4_t x_reg4 = vld1q_f32(&X(i + 4));

        // 每行起始地址，按循环指定的stride跳跃
        float32x4_t w_l00_reg = vld1q_f32(&W(0, i));
        y_l00_sum = vfmaq_f32(y_l00_sum, w_l00_reg, x_reg0);
        float32x4_t w_l04_reg = vld1q_f32(&W(0, i + 4));
        y_l04_sum = vfmaq_f32(y_l04_sum, w_l04_reg, x_reg4);

        float32x4_t w_l10_reg = vld1q_f32(&W(1, i));
        y_l10_sum = vfmaq_f32(y_l10_sum, w_l10_reg, x_reg0);
        float32x4_t w_l14_reg = vld1q_f32(&W(1, i + 4));
        y_l14_sum = vfmaq_f32(y_l14_sum, w_l14_reg, x_reg4);

        float32x4_t w_l20_reg = vld1q_f32(&W(2, i));
        y_l20_sum = vfmaq_f32(y_l20_sum, w_l20_reg, x_reg0);
        float32x4_t w_l24_reg = vld1q_f32(&W(2, i + 4));
        y_l24_sum = vfmaq_f32(y_l24_sum, w_l24_reg, x_reg4);

        float32x4_t w_l30_reg = vld1q_f32(&W(3, i));
        y_l30_sum = vfmaq_f32(y_l30_sum, w_l30_reg, x_reg0);
        float32x4_t w_l34_reg = vld1q_f32(&W(3, i + 4));
        y_l34_sum = vfmaq_f32(y_l34_sum, w_l34_reg, x_reg4);

        // reuse registers
        w_l00_reg = vld1q_f32(&W(4, i));
        y_l40_sum = vfmaq_f32(y_l40_sum, w_l00_reg, x_reg0);
        w_l04_reg = vld1q_f32(&W(4, i + 4));
        y_l44_sum = vfmaq_f32(y_l44_sum, w_l04_reg, x_reg4);

        w_l10_reg = vld1q_f32(&W(5, i));
        y_l50_sum = vfmaq_f32(y_l50_sum, w_l10_reg, x_reg0);
        w_l14_reg = vld1q_f32(&W(5, i + 4));
        y_l54_sum = vfmaq_f32(y_l54_sum, w_l14_reg, x_reg4);

        w_l20_reg = vld1q_f32(&W(6, i));
        y_l60_sum = vfmaq_f32(y_l60_sum, w_l20_reg, x_reg0);
        w_l24_reg = vld1q_f32(&W(6, i + 4));
        y_l64_sum = vfmaq_f32(y_l64_sum, w_l24_reg, x_reg4);

        w_l30_reg = vld1q_f32(&W(7, i));
        y_l70_sum = vfmaq_f32(y_l70_sum, w_l30_reg, x_reg0);
        w_l34_reg = vld1q_f32(&W(7, i + 4));
        y_l74_sum = vfmaq_f32(y_l74_sum, w_l34_reg, x_reg4);
    }
    if (i != n) {
        printf("%s:%d %d != %d\n", __FILE__, __LINE__, i, n);
    }

    // float* y_ptr = xout;
    // float32x4_t tmp_sum = vaddq_f32(y_l00_sum, y_l04_sum);
    // *y_ptr++ = vaddvq_f32(tmp_sum);
    // tmp_sum = vaddq_f32(y_l10_sum, y_l14_sum);
    // *y_ptr++ = vaddvq_f32(tmp_sum);
    // tmp_sum = vaddq_f32(y_l20_sum, y_l24_sum);
    // *y_ptr++ = vaddvq_f32(tmp_sum);
    // tmp_sum = vaddq_f32(y_l30_sum, y_l34_sum);
    // *y_ptr++ = vaddvq_f32(tmp_sum);
    // tmp_sum = vaddq_f32(y_l40_sum, y_l44_sum);
    // *y_ptr++ = vaddvq_f32(tmp_sum);
    // tmp_sum = vaddq_f32(y_l50_sum, y_l54_sum);
    // *y_ptr++ = vaddvq_f32(tmp_sum);
    // tmp_sum = vaddq_f32(y_l60_sum, y_l64_sum);
    // *y_ptr++ = vaddvq_f32(tmp_sum);
    // tmp_sum = vaddq_f32(y_l70_sum, y_l74_sum);
    // *y_ptr++ = vaddvq_f32(tmp_sum);

    float* y_ptr = xout;
    float32x4_t y_reg0 = vld1q_f32(y_ptr);
    float32x4_t y_reg4 = vld1q_f32(y_ptr + 4);
    float32x4_t zeros = vdupq_n_f32(0.0f);
    float32x4_t ones = vdupq_n_f32(1.0f);

    float32_t y_sum = vaddvq_f32(vaddq_f32(y_l00_sum, y_l04_sum));
    float32x4_t mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 0);
    float32x4_t y_res0 = vfmaq_n_f32(y_reg0, mask, y_sum);

    y_sum = vaddvq_f32(vaddq_f32(y_l10_sum, y_l14_sum));
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 1);
    y_res0 = vfmaq_n_f32(y_res0, mask, y_sum);
    y_sum = vaddvq_f32(vaddq_f32(y_l20_sum, y_l24_sum));
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 2);
    y_res0 = vfmaq_n_f32(y_res0, mask, y_sum);
    y_sum = vaddvq_f32(vaddq_f32(y_l30_sum, y_l34_sum));
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 3);
    y_res0 = vfmaq_n_f32(y_res0, mask, y_sum);
    vst1q_f32(y_ptr, y_res0);

    y_sum = vaddvq_f32(vaddq_f32(y_l40_sum, y_l44_sum));
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 0);
    float32x4_t y_res1 = vfmaq_n_f32(y_reg4, mask, y_sum);
    y_sum = vaddvq_f32(vaddq_f32(y_l50_sum, y_l54_sum));
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 1);
    y_res1 = vfmaq_n_f32(y_res1, mask, y_sum);
    y_sum = vaddvq_f32(vaddq_f32(y_l60_sum, y_l64_sum));
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 2);
    y_res1 = vfmaq_n_f32(y_res1, mask, y_sum);
    y_sum = vaddvq_f32(vaddq_f32(y_l70_sum, y_l74_sum));
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 3);
    y_res1 = vfmaq_n_f32(y_res1, mask, y_sum);
    vst1q_f32(y_ptr + 4, y_res1);
}

void AddDot4x8(int n, float* xout, float* x, float* w, int ldw) {
    int i;

    // 先暂存循环展开的每行数据，整行处理完得到最终结果再存到内存。
    // 注：也可以每次写入xout，下一个循环再读出来累加，但效率不高
    float32x4_t y_l00_sum = {0};
    float32x4_t y_l04_sum = {0};
    float32x4_t y_l10_sum = {0};
    float32x4_t y_l14_sum = {0};
    float32x4_t y_l20_sum = {0};
    float32x4_t y_l24_sum = {0};
    float32x4_t y_l30_sum = {0};
    float32x4_t y_l34_sum = {0};

    for (i = 0; i < n; i += 8) {
        if (i + 8 > n)
            break;

        // 加载 X 数据
        float32x4_t x_reg0 = vld1q_f32(&X(i));
        float32x4_t x_reg4 = vld1q_f32(&X(i + 4));

        // 每行起始地址，按循环指定的stride跳跃
        float32x4_t w_l00_reg = vld1q_f32(&W(0, i));
        y_l00_sum = vfmaq_f32(y_l00_sum, w_l00_reg, x_reg0);
        float32x4_t w_l04_reg = vld1q_f32(&W(0, i + 4));
        y_l04_sum = vfmaq_f32(y_l04_sum, w_l04_reg, x_reg4);

        float32x4_t w_l10_reg = vld1q_f32(&W(1, i));
        y_l10_sum = vfmaq_f32(y_l10_sum, w_l10_reg, x_reg0);
        float32x4_t w_l14_reg = vld1q_f32(&W(1, i + 4));
        y_l14_sum = vfmaq_f32(y_l14_sum, w_l14_reg, x_reg4);

        float32x4_t w_l20_reg = vld1q_f32(&W(2, i));
        y_l20_sum = vfmaq_f32(y_l20_sum, w_l20_reg, x_reg0);
        float32x4_t w_l24_reg = vld1q_f32(&W(2, i + 4));
        y_l24_sum = vfmaq_f32(y_l24_sum, w_l24_reg, x_reg4);

        float32x4_t w_l30_reg = vld1q_f32(&W(3, i));
        y_l30_sum = vfmaq_f32(y_l30_sum, w_l30_reg, x_reg0);
        float32x4_t w_l34_reg = vld1q_f32(&W(3, i + 4));
        y_l34_sum = vfmaq_f32(y_l34_sum, w_l34_reg, x_reg4);
    }
    if (i != n) {
        printf("%s:%d %d != %d\n", __FILE__, __LINE__, i, n);
    }

    // float* y_ptr = xout;
    // float32x4_t tmp_l0_sum = vaddq_f32(y_l00_sum, y_l04_sum);
    // *y_ptr++ = vaddvq_f32(tmp_l0_sum);
    // float32x4_t tmp_l1_sum = vaddq_f32(y_l10_sum, y_l14_sum);
    // *y_ptr++ = vaddvq_f32(tmp_l1_sum);
    // float32x4_t tmp_l2_sum = vaddq_f32(y_l20_sum, y_l24_sum);
    // *y_ptr++ = vaddvq_f32(tmp_l2_sum);
    // float32x4_t tmp_l3_sum = vaddq_f32(y_l30_sum, y_l34_sum);
    // *y_ptr++ = vaddvq_f32(tmp_l3_sum);

    float* y_ptr = xout;
    float32x4_t y_reg0 = vld1q_f32(y_ptr);
    // float32x4_t y_reg4 = vld1q_f32(y_ptr + 4);
    float32x4_t zeros = vdupq_n_f32(0.0f);
    float32x4_t ones = vdupq_n_f32(1.0f);

    float32_t y_sum = vaddvq_f32(vaddq_f32(y_l00_sum, y_l04_sum));
    float32x4_t mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 0);
    float32x4_t y_res0 = vfmaq_n_f32(y_reg0, mask, y_sum);

    y_sum = vaddvq_f32(vaddq_f32(y_l10_sum, y_l14_sum));
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 1);
    y_res0 = vfmaq_n_f32(y_res0, mask, y_sum);
    y_sum = vaddvq_f32(vaddq_f32(y_l20_sum, y_l24_sum));
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 2);
    y_res0 = vfmaq_n_f32(y_res0, mask, y_sum);
    y_sum = vaddvq_f32(vaddq_f32(y_l30_sum, y_l34_sum));
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 3);
    y_res0 = vfmaq_n_f32(y_res0, mask, y_sum);
    vst1q_f32(y_ptr, y_res0);
}

void AddDot4x16(int n, float* xout, float* x, float* w, int ldw) {
    int i;

    // 先暂存循环展开的每行数据，整行处理完得到最终结果再存到内存。
    // 注：也可以每次写入xout，下一个循环再读出来累加，但效率不高
    float32x4_t y_l00_sum = {0};
    float32x4_t y_l04_sum = {0};
    float32x4_t y_l08_sum = {0};
    float32x4_t y_l0c_sum = {0};
    float32x4_t y_l10_sum = {0};
    float32x4_t y_l14_sum = {0};
    float32x4_t y_l18_sum = {0};
    float32x4_t y_l1c_sum = {0};
    float32x4_t y_l20_sum = {0};
    float32x4_t y_l24_sum = {0};
    float32x4_t y_l28_sum = {0};
    float32x4_t y_l2c_sum = {0};
    float32x4_t y_l30_sum = {0};
    float32x4_t y_l34_sum = {0};
    float32x4_t y_l38_sum = {0};
    float32x4_t y_l3c_sum = {0};

    for (i = 0; i < n; i += 16) {
        if (i + 16 > n)
            break;

        // 加载 X 数据
        float32x4_t x_reg0 = vld1q_f32(&X(i));
        float32x4_t x_reg4 = vld1q_f32(&X(i + 4));
        float32x4_t x_reg8 = vld1q_f32(&X(i + 8));
        float32x4_t x_regc = vld1q_f32(&X(i + 12));

        // 每行起始地址，按循环指定的stride跳跃
        float32x4_t w_l00_reg = vld1q_f32(&W(0, i));
        y_l00_sum = vfmaq_f32(y_l00_sum, w_l00_reg, x_reg0);
        float32x4_t w_l04_reg = vld1q_f32(&W(0, i + 4));
        y_l04_sum = vfmaq_f32(y_l04_sum, w_l04_reg, x_reg4);
        float32x4_t w_l08_reg = vld1q_f32(&W(0, i + 8));
        y_l08_sum = vfmaq_f32(y_l08_sum, w_l08_reg, x_reg8);
        float32x4_t w_l0c_reg = vld1q_f32(&W(0, i + 12));
        y_l0c_sum = vfmaq_f32(y_l0c_sum, w_l0c_reg, x_regc);

        float32x4_t w_l10_reg = vld1q_f32(&W(1, i));
        y_l10_sum = vfmaq_f32(y_l10_sum, w_l10_reg, x_reg0);
        float32x4_t w_l14_reg = vld1q_f32(&W(1, i + 4));
        y_l14_sum = vfmaq_f32(y_l14_sum, w_l14_reg, x_reg4);
        float32x4_t w_l18_reg = vld1q_f32(&W(1, i + 8));
        y_l18_sum = vfmaq_f32(y_l18_sum, w_l18_reg, x_reg8);
        float32x4_t w_l1c_reg = vld1q_f32(&W(1, i + 12));
        y_l1c_sum = vfmaq_f32(y_l1c_sum, w_l1c_reg, x_regc);

        // reuse regiter
        float32x4_t w_l20_reg = vld1q_f32(&W(2, i));
        y_l20_sum = vfmaq_f32(y_l20_sum, w_l20_reg, x_reg0);
        float32x4_t w_l24_reg = vld1q_f32(&W(2, i + 4));
        y_l24_sum = vfmaq_f32(y_l24_sum, w_l24_reg, x_reg4);
        float32x4_t w_l28_reg = vld1q_f32(&W(2, i + 8));
        y_l28_sum = vfmaq_f32(y_l28_sum, w_l28_reg, x_reg8);
        float32x4_t w_l2c_reg = vld1q_f32(&W(2, i + 12));
        y_l2c_sum = vfmaq_f32(y_l2c_sum, w_l2c_reg, x_regc);

        float32x4_t w_l30_reg = vld1q_f32(&W(3, i));
        y_l30_sum = vfmaq_f32(y_l30_sum, w_l30_reg, x_reg0);
        float32x4_t w_l34_reg = vld1q_f32(&W(3, i + 4));
        y_l34_sum = vfmaq_f32(y_l34_sum, w_l34_reg, x_reg4);
        float32x4_t w_l38_reg = vld1q_f32(&W(3, i + 8));
        y_l38_sum = vfmaq_f32(y_l38_sum, w_l38_reg, x_reg8);
        float32x4_t w_l3c_reg = vld1q_f32(&W(3, i + 12));
        y_l3c_sum = vfmaq_f32(y_l3c_sum, w_l3c_reg, x_regc);
    }
    if (i != n) {
        printf("%s:%d %d != %d\n", __FILE__, __LINE__, i, n);
    }

    // float* y_ptr = xout;
    // float32x4_t temp04 = vaddq_f32(y_l00_sum, y_l04_sum);
    // float32x4_t temp0 = vaddq_f32(temp04, y_l08_sum);
    // *y_ptr++ = vaddvq_f32(temp0);

    // temp04 = vaddq_f32(y_l10_sum, y_l14_sum);
    // temp0 = vaddq_f32(temp04, y_l18_sum);
    // *y_ptr++ = vaddvq_f32(temp0);

    // temp04 = vaddq_f32(y_l20_sum, y_l24_sum);
    // temp0 = vaddq_f32(temp04, y_l28_sum);
    // *y_ptr++ = vaddvq_f32(temp0);

    // temp04 = vaddq_f32(y_l30_sum, y_l34_sum);
    // temp0 = vaddq_f32(temp04, y_l38_sum);
    // *y_ptr++ = vaddvq_f32(temp0);

    float* y_ptr = xout;
    float32x4_t y_reg0 = vld1q_f32(y_ptr);
    // float32x4_t y_reg4 = vld1q_f32(y_ptr + 4);
    float32x4_t zeros = vdupq_n_f32(0.0f);
    float32x4_t ones = vdupq_n_f32(1.0f);

    float32_t y_sum = vaddvq_f32(vaddq_f32(vaddq_f32(y_l00_sum, y_l04_sum),
                                           vaddq_f32(y_l08_sum, y_l0c_sum)));
    float32x4_t mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 0);
    float32x4_t y_res0 = vfmaq_n_f32(y_reg0, mask, y_sum);

    y_sum = vaddvq_f32(vaddq_f32(vaddq_f32(y_l10_sum, y_l14_sum),
                                 vaddq_f32(y_l18_sum, y_l1c_sum)));
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 1);
    y_res0 = vfmaq_n_f32(y_res0, mask, y_sum);
    y_sum = vaddvq_f32(vaddq_f32(vaddq_f32(y_l20_sum, y_l24_sum),
                                 vaddq_f32(y_l28_sum, y_l2c_sum)));
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 2);
    y_res0 = vfmaq_n_f32(y_res0, mask, y_sum);
    y_sum = vaddvq_f32(vaddq_f32(vaddq_f32(y_l30_sum, y_l34_sum),
                                 vaddq_f32(y_l38_sum, y_l3c_sum)));
    mask = vsetq_lane_f32(vgetq_lane_f32(ones, 0), zeros, 3);
    y_res0 = vfmaq_n_f32(y_res0, mask, y_sum);
    vst1q_f32(y_ptr, y_res0);
}
