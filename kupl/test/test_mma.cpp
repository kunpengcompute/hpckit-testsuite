#if defined(ENABLE_KUPL_MMA)

#include <arm_bf16.h>
#include <cstdint>
#include "gtest/gtest.h"
#include "kupl_mma.h"
using namespace kupl::tensor;

#if defined(__clang__)
#define test_kupl_za
#define test_kupl_streaming
#elif defined(__GNUC__)
#define test_kupl_za        __arm_new("za")
#define test_kupl_streaming __arm_streaming
#endif

test_kupl_za
void mma_32x16x1_F64F64F64_kernel(double *data_a, double *data_b, double *data_c) test_kupl_streaming
{
    auto shape_a = make_shape(Int<32>{}, Int<512>{});
    auto shape_b = make_shape(Int<512>{}, Int<16>{});
    auto shape_c = make_shape(Int<32>{}, Int<16>{});

    auto stride_a = make_stride(Int<1>{}, Int<32>{});
    auto stride_b = make_stride(Int<16>{}, Int<1>{});
    auto stride_c = make_stride(Int<16>{}, Int<1>{});

    auto layout_a = make_layout(shape_a, stride_a);
    auto layout_b = make_layout(shape_b, stride_b);
    auto layout_c = make_layout(shape_c, stride_c);

    auto atom_mma_shape = make_shape(Int<1>{}, Int<1>{}, Int<512>{});
    auto tiled_mma = make_tiled_mma(Ops<KP36_32x16x1_F64F64F64>{}, atom_mma_shape);
    auto atom_store_shape = make_shape(Int<1>{}, Int<1>{});
    auto tiled_store = make_tiled_store(Ops<KP36_32x16_F64_STORE>{}, atom_store_shape);

    auto tensor_a = make_tensor(data_a, layout_a);
    auto tensor_b = make_tensor(data_b, layout_b);
    auto tensor_c = make_tensor(data_c, layout_c);
    mma(tiled_mma, tensor_c, tensor_a, tensor_b, tensor_c);
    store(tiled_store, tensor_c);
}

TEST(test_mma, kupl_mma_001)
{
    const int M = 32;
    const int N = 16;
    const int K = 512;
    double *data_a = (double*)malloc(sizeof(double) * M * K);
    double *data_b = (double*)malloc(sizeof(double) * K * N);
    double *data_c = (double*)malloc(sizeof(double) * M * N);
    for (int i = 0; i < K; i++) {
        for(int j = 0; j < M; j++) {
            if (i == j) {
                data_a[i * M + j] = 1.0;
            } else {
                data_a[i * M + j] = 0.0;
            }
        }
    }
    for (int i = 0; i < K; i++) {
        for(int j = 0; j < N; j++) {
            data_b[i * N + j] = (double)(i * N + j);
        }
    }
    for (int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            data_c[i * N + j] = 0.0;
        }
    }

    mma_32x16x1_F64F64F64_kernel(data_a, data_b, data_c);

    bool res = true;
    for (int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            if (data_c[i * N + j] != i * N + j) {
                res = false;
            }
        }
    }
    ASSERT_TRUE(res);
    free(data_c);
    free(data_b);
    free(data_a);
}

#endif