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

#define W(i, j) w[(i)*n + (j)]
#define X(i) x[(i)]
#define Y(i) xout[(i)]

static int unroll_num = 8;
static int vec_size = 4;

void matmul_naive(float* xout, float* x, float* w, int n, int d) {
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
    matmul_naive(ans, x, w, n, d);
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

// 在K维度上，循环展开一次计算8行，向量化4个连续元素
void AddDot8x4(int, float*, float*, float*);
// 在K维度上，循环展开一次计算8行，向量化8个连续元素
void AddDot8x8(int, float*, float*, float*);
// 在K维度上，循环展开一次计算8行，向量化8个连续元素
void AddDot4x8(int, float*, float*, float*);
// 在K维度上，循环展开一次计算4行，向量化16个连续元素
void AddDot4x16(int, float*, float*, float*);


void matmul_unroll(float* xout, float* x, float* w, int n, int d) {
    int i = 0;
    
    // 循环展开，一次处理的行数
    if (unroll_num == 8) {
        for (i = 0; i < d; i += 8) {
            if (i + 8 > d)
                break;
            if (vec_size == 4)
                AddDot8x4(n, &Y(i), &X(0), &W(i, 0));
            else if (vec_size == 8)
                AddDot8x8(n, &Y(i), &X(0), &W(i, 0));
            else
                fprintf(stderr, "unsupported AddDot8x%d\n", vec_size);
        }
    } else if (unroll_num == 4) {
        for (i = 0; i < d; i += 4) {
            if (i + 4 > d)
                break;
            if (vec_size == 8)
                AddDot4x8(n, &Y(i), &X(0), &W(i, 0));
            else if (vec_size == 16)
                AddDot4x16(n, &Y(i), &X(0), &W(i, 0));
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

void AddDot8x4(int n, float* xout, float* x, float* w) {
    int i;

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

    float* y_ptr = xout;
    float32x2_t temp_l0 =
            vpadd_f32(vget_low_f32(y_l00_sum), vget_high_f32(y_l00_sum));
    *y_ptr++ = vpadds_f32(temp_l0);

    float32x2_t temp_l1 =
            vpadd_f32(vget_low_f32(y_l10_sum), vget_high_f32(y_l10_sum));
    *y_ptr++ = vpadds_f32(temp_l1);

    float32x2_t temp_l2 =
            vpadd_f32(vget_low_f32(y_l20_sum), vget_high_f32(y_l20_sum));
    *y_ptr++ = vpadds_f32(temp_l2);

    float32x2_t temp_l3 =
            vpadd_f32(vget_low_f32(y_l30_sum), vget_high_f32(y_l30_sum));
    *y_ptr++ = vpadds_f32(temp_l3);

    float32x2_t temp_l4 =
            vpadd_f32(vget_low_f32(y_l40_sum), vget_high_f32(y_l40_sum));
    *y_ptr++ = vpadds_f32(temp_l4);

    float32x2_t temp_l5 =
            vpadd_f32(vget_low_f32(y_l50_sum), vget_high_f32(y_l50_sum));
    *y_ptr++ = vpadds_f32(temp_l5);

    float32x2_t temp_l6 =
            vpadd_f32(vget_low_f32(y_l60_sum), vget_high_f32(y_l60_sum));
    *y_ptr++ = vpadds_f32(temp_l6);

    float32x2_t temp_l7 =
            vpadd_f32(vget_low_f32(y_l70_sum), vget_high_f32(y_l70_sum));
    *y_ptr++ = vpadds_f32(temp_l7);
}

void AddDot8x8(int n, float* xout, float* x, float* w) {
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

        float32x4_t w_l40_reg = vld1q_f32(&W(4, i));
        y_l40_sum = vfmaq_f32(y_l40_sum, w_l40_reg, x_reg0);
        float32x4_t w_l44_reg = vld1q_f32(&W(4, i + 4));
        y_l44_sum = vfmaq_f32(y_l44_sum, w_l44_reg, x_reg4);

        float32x4_t w_l50_reg = vld1q_f32(&W(5, i));
        y_l50_sum = vfmaq_f32(y_l50_sum, w_l50_reg, x_reg0);
        float32x4_t w_l54_reg = vld1q_f32(&W(5, i + 4));
        y_l54_sum = vfmaq_f32(y_l54_sum, w_l54_reg, x_reg4);

        float32x4_t w_l60_reg = vld1q_f32(&W(6, i));
        y_l60_sum = vfmaq_f32(y_l60_sum, w_l60_reg, x_reg0);
        float32x4_t w_l64_reg = vld1q_f32(&W(6, i + 4));
        y_l64_sum = vfmaq_f32(y_l64_sum, w_l64_reg, x_reg4);

        float32x4_t w_l70_reg = vld1q_f32(&W(7, i));
        y_l70_sum = vfmaq_f32(y_l70_sum, w_l70_reg, x_reg0);
        float32x4_t w_l74_reg = vld1q_f32(&W(7, i + 4));
        y_l74_sum = vfmaq_f32(y_l74_sum, w_l74_reg, x_reg4);
    }
    if (i != n) {
        printf("%s:%d %d != %d\n", __FILE__, __LINE__, i, n);
    }

    float* y_ptr = xout;
    float32x4_t tmp_sum = vaddq_f32(y_l00_sum, y_l04_sum);
    float32x2_t tmp = vpadd_f32(vget_low_f32(tmp_sum), vget_high_f32(tmp_sum));
    *y_ptr++ = vpadds_f32(tmp);

    tmp_sum = vaddq_f32(y_l10_sum, y_l14_sum);
    tmp = vpadd_f32(vget_low_f32(tmp_sum), vget_high_f32(tmp_sum));
    *y_ptr++ = vpadds_f32(tmp);

    tmp_sum = vaddq_f32(y_l20_sum, y_l24_sum);
    tmp = vpadd_f32(vget_low_f32(tmp_sum), vget_high_f32(tmp_sum));
    *y_ptr++ = vpadds_f32(tmp);

    tmp_sum = vaddq_f32(y_l30_sum, y_l34_sum);
    tmp = vpadd_f32(vget_low_f32(tmp_sum), vget_high_f32(tmp_sum));
    *y_ptr++ = vpadds_f32(tmp);

    tmp_sum = vaddq_f32(y_l40_sum, y_l44_sum);
    tmp = vpadd_f32(vget_low_f32(tmp_sum), vget_high_f32(tmp_sum));
    *y_ptr++ = vpadds_f32(tmp);

    tmp_sum = vaddq_f32(y_l50_sum, y_l54_sum);
    tmp = vpadd_f32(vget_low_f32(tmp_sum), vget_high_f32(tmp_sum));
    *y_ptr++ = vpadds_f32(tmp);

    tmp_sum = vaddq_f32(y_l60_sum, y_l64_sum);
    tmp = vpadd_f32(vget_low_f32(tmp_sum), vget_high_f32(tmp_sum));
    *y_ptr++ = vpadds_f32(tmp);

    tmp_sum = vaddq_f32(y_l70_sum, y_l74_sum);
    tmp = vpadd_f32(vget_low_f32(tmp_sum), vget_high_f32(tmp_sum));
    *y_ptr++ = vpadds_f32(tmp);
}

void AddDot4x8(int n, float* xout, float* x, float* w) {
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
        float32x4_t x_reg4 = vld1q_f32(&X(i+4));

        // 每行起始地址，按循环指定的stride跳跃
        float32x4_t w_l00_reg = vld1q_f32(&W(0, i));
        y_l00_sum = vfmaq_f32(y_l00_sum, w_l00_reg, x_reg0);
        float32x4_t w_l04_reg = vld1q_f32(&W(0, i+4));
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

    float* y_ptr = xout;
    float32x4_t tmp_l0_sum = vaddq_f32(y_l00_sum, y_l04_sum);
    float32x2_t tmp_l0 =
            vpadd_f32(vget_low_f32(tmp_l0_sum), vget_high_f32(tmp_l0_sum));
    *y_ptr++ = vpadds_f32(tmp_l0);

    float32x4_t tmp_l1_sum = vaddq_f32(y_l10_sum, y_l14_sum);
    float32x2_t tmp_l1 =
            vpadd_f32(vget_low_f32(tmp_l1_sum), vget_high_f32(tmp_l1_sum));
    *y_ptr++ = vpadds_f32(tmp_l1);

    float32x4_t tmp_l2_sum = vaddq_f32(y_l20_sum, y_l24_sum);
    float32x2_t tmp_l2 =
            vpadd_f32(vget_low_f32(tmp_l2_sum), vget_high_f32(tmp_l2_sum));
    *y_ptr++ = vpadds_f32(tmp_l2);

    float32x4_t tmp_l3_sum = vaddq_f32(y_l30_sum, y_l34_sum);
    float32x2_t tmp_l3 =
            vpadd_f32(vget_low_f32(tmp_l3_sum), vget_high_f32(tmp_l3_sum));
    *y_ptr++ = vpadds_f32(tmp_l3);
}


void AddDot4x16(int n, float* xout, float* x, float* w) {
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
        float32x4_t x_reg4 = vld1q_f32(&X(i+4));
        float32x4_t x_reg8 = vld1q_f32(&X(i+8));
        float32x4_t x_regc = vld1q_f32(&X(i+12));

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

    float* y_ptr = xout;
    float32x4_t temp04 = vaddq_f32(y_l00_sum, y_l04_sum);
    float32x4_t temp8c = vaddq_f32(y_l08_sum, y_l0c_sum);
    float32x4_t temp0 = vaddq_f32(temp04, temp8c);
    float32x2_t temp1 = vpadd_f32(vget_low_f32(temp0), vget_high_f32(temp0));
    *y_ptr++ = vpadds_f32(temp1);

    temp04 = vaddq_f32(y_l10_sum, y_l14_sum);
    temp8c = vaddq_f32(y_l18_sum, y_l1c_sum);
    temp0 = vaddq_f32(temp04, temp8c);
    temp1 = vpadd_f32(vget_low_f32(temp0), vget_high_f32(temp0));
    *y_ptr++ = vpadds_f32(temp1);

    temp04 = vaddq_f32(y_l20_sum, y_l24_sum);
    temp8c = vaddq_f32(y_l28_sum, y_l2c_sum);
    temp0 = vaddq_f32(temp04, temp8c);
    temp1 = vpadd_f32(vget_low_f32(temp0), vget_high_f32(temp0));
    *y_ptr++ = vpadds_f32(temp1);

    temp04 = vaddq_f32(y_l30_sum, y_l34_sum);
    temp8c = vaddq_f32(y_l38_sum, y_l3c_sum);
    temp0 = vaddq_f32(temp04, temp8c);
    temp1 = vpadd_f32(vget_low_f32(temp0), vget_high_f32(temp0));
    *y_ptr++ = vpadds_f32(temp1);
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
    matmul_unroll(y, x, w, n, d);

    float max_err = check(y, x, w, n, d);
    printf("max_err:%f\n", max_err);
    if (max_err > 0.1f) {
        mperf_throw(mperf::MperfError, "ERROR: result check error.");
    }

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
            matmul_unroll(y, x, w, n, d);
        }
        mpf_tma.sample_and_stop(iter_num);
    }

    for (size_t i = 0; i < uncore_evt_num; ++i) {
        mpf_tma.start_uncore(i);
        for (size_t j = 0; j < iter_num; ++j) {
            matmul_unroll(y, x, w, n, d);

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
                "./llama2_cpu_matmul_unroll core_id unroll_rows vec_size\n");
        return -1;
    }
    const int dev_id = atoi(argv[1]);
    unroll_num = atoi(argv[2]);
    vec_size = atoi(argv[3]);

    printf("unroll: %d lines \nvectorize: %d floats\n", unroll_num, vec_size);

    if (set_cpu_thread_affinity_spec_core(dev_id)) {
        printf("faild set thread affinity(core %d)\n", dev_id);
    }

    gettma(128, 128);
    gettma(288, 288);
    gettma(512, 512);

    return 0;
}
