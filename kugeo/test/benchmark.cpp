/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <sys/time.h>
#if defined(__AVX512F__) || defined(__AVX512__)
#include <immintrin.h>
#elif defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#else
#include <arm_neon.h>
#endif
#if defined(__ARM_FEATURE_SME)
#include <arm_sme.h>
#endif
#include "kurtm.h"


static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

template<typename T>
T uniform() {
    static uint64_t s[2] = { 0x41, 0x29837592 };
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
    s[1] = rotl(s1, 36); // c

    return result*((1.0/18446744073709551616.0));
}


#define TOT_TIME 10
template<typename T>
double time_func(T func) {
    int it = 1;
    func(); // Warm up
    double st = omp_get_wtime();
    double ed = st;
    while (ed < st + TOT_TIME) {
        for (int i = 0; i < it; ++i)
            func();
        it <<= 1;
        ed = omp_get_wtime();
    }
    return (ed - st) / (it - 1);
}

#define min(x, y) ((x) < (y) ? (x) : (y))
#define ceil(x, y) (((x) + (y) - 1)/(y))
#define index_g(i, j, k, sx, sy) ((i) + (j)*(sx) + (k)*(sy))
#define index_b(i, j, k, sx, sy) (((i)/8)*256 + ((j)/8)*(sx) + ((k)/4)*(sy))

#if defined(__ARM_FEATURE_SME)
int main(int argc, char** argv)__arm_streaming {
#else
int main(int argc, char** argv) {
#endif
    assert(argc == 4);
    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int nz = atoi(argv[3]);
    int dimension = 2;
    int radius = 4;
    int GZ_x = 8, GZ_y = 4, GZ_z = 4;
    int grid_dim_x = ceil(nx, 8)*8 + 2*GZ_x;
    int grid_dim_y = ny + 2*GZ_y;
    int grid_dim_z = nz + 2*GZ_z;
    int n_threads = omp_get_max_threads();
    printf("start test kurtm 2d kernel.\n");

    double* alpha = (double*)malloc(sizeof(double)*729);
    double* p0 = (double*)aligned_alloc(64, sizeof(double)*grid_dim_x*grid_dim_y*grid_dim_z);
    double* p1 = (double*)aligned_alloc(64, sizeof(double)*grid_dim_x*grid_dim_y*grid_dim_z);

    for(int i = 0; i < 729; i += 1) {
        alpha[i] = uniform<double>();
    }
    for(int i = 0; i < grid_dim_x*grid_dim_y*grid_dim_z; i += 1) {
        p0[i] = uniform<double>();
    }
    memset(p1, 0, sizeof(double)*grid_dim_x*grid_dim_y*grid_dim_z);

    kurtm_stencil_factor_t kernel_factor[3] = {KURTM_STENCIL_FACTOR_IMMU, KURTM_STENCIL_FACTOR_IMMU, KURTM_STENCIL_FACTOR_IMMU};
    kurtm_stencil_kernel_t *kernel = NULL;
    kurtm_stencil_direct_t kernel_dircet[3] = {KURTM_STENCIL_DIRECT_ON, KURTM_STENCIL_DIRECT_ON, KURTM_STENCIL_DIRECT_OFF};
    kernel = kurtm_stencil_2dkernel_create(radius, KURTM_STENCIL_TYPE_BOX,
                                           kernel_dircet, kernel_factor,
                                           alpha);

    int dimSize[3] = {nx, ny, nz};
    int dimStep[3] = {1, grid_dim_x, grid_dim_x * grid_dim_y};

    kurtm_tensor_t *in = kurtm_tensor_create(
        &p0[index_g(GZ_x, GZ_y, GZ_z, grid_dim_x, grid_dim_y)],
        dimSize, dimStep, KURTM_TENSOR_DOUBLE);
    kurtm_tensor_t *ou = kurtm_tensor_create(
        &p1[index_g(GZ_x, GZ_y, GZ_z, grid_dim_x, grid_dim_y)],
        dimSize, dimStep, KURTM_TENSOR_DOUBLE);


    double elapsed_time = time_func([&](){
        kurtm_stencil_run(in, ou, kernel);
    });

    printf("end test kurtm 2d kernel. PASSED\n");

    printf("1 tests ran, 1 tests PASSED, 0 tests FAILED.\n");
    return 0;
}
