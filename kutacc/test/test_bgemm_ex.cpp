#undef NEEDBUNDERSCORE

#include <arm_bf16.h>
#include <arm_neon.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "kutacc_core.h"

#define MATRIX_ROW 31
#define MATRIX_COL 15
#define BF16_EPS 5.0

static uint32_t tests_count = 0;
static uint32_t passed_count = 0;
static uint32_t failed_count = 0;

static void cal_leading_dimension(char transA, char transB, BLASINT *lda, BLASINT *ldb)
{
    BLASINT m = MATRIX_ROW;
    BLASINT n = MATRIX_ROW;
    BLASINT k = MATRIX_COL;

    if (transA == 'N') {
        *lda = m;
    } else {
        *lda = k;
    }
    if (transB == 'N') {
        *ldb = k;
    } else {
        *ldb = n;
    }
}

static void init_expect_c_matrix(char transA, char transB, BLASINT m, BLASINT n, BLASINT k, __bf16 alpha, __bf16 beta,
    __bf16 bgemm_a[], __bf16 bgemm_b[], __bf16 expect_c[])
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            float cij = 0;
            for (int t = 0; t < k; t++) {
                float aik = (transA == 'N') ? vcvtah_f32_bf16(bgemm_a[i + t * MATRIX_ROW]) :
                                              vcvtah_f32_bf16(bgemm_a[t + i * MATRIX_COL]);
                float bkj = (transB == 'N') ? vcvtah_f32_bf16(bgemm_b[t + j * MATRIX_COL]) :
                                              vcvtah_f32_bf16(bgemm_b[j + t * MATRIX_ROW]);
                cij += aik * bkj;
            }
            expect_c[i + j * MATRIX_ROW] = vcvth_bf16_f32(
                vcvtah_f32_bf16(alpha) * cij + vcvtah_f32_bf16(beta) * vcvtah_f32_bf16(expect_c[i + j * MATRIX_ROW]));
        }
    }
}

static bool test_bgemm_ex(char transA, char transB, uint32_t prepack_mask)
{
    __bf16 bgemm_a[MATRIX_ROW * MATRIX_COL];
    __bf16 bgemm_b[MATRIX_COL * MATRIX_ROW];
    __bf16 bgemm_c[MATRIX_ROW * MATRIX_ROW];
    __bf16 expect_c[MATRIX_ROW * MATRIX_ROW];
    __bf16 *sa = NULL;
    __bf16 *sb = NULL;

    for (int i = 0; i < MATRIX_ROW * MATRIX_COL; i++) {
        bgemm_a[i] = vcvth_bf16_f32(1 + 0.01 * i);
        bgemm_b[i] = vcvth_bf16_f32(1 + 0.01 * i);
    }
    for (int i = 0; i < MATRIX_ROW * MATRIX_ROW; i++) {
        bgemm_c[i] = vcvth_bf16_f32(1 + 0.01 * i);
        expect_c[i] = vcvth_bf16_f32(1 + 0.01 * i);
    }

    BLASINT m = MATRIX_ROW;
    BLASINT n = MATRIX_ROW;
    BLASINT k = MATRIX_COL;
    BLASINT lda = m;
    BLASINT ldb = n;
    BLASINT ldc = m;
    __bf16 alpha = vcvth_bf16_f32(1.0);
    __bf16 beta = vcvth_bf16_f32(3.0);

    init_expect_c_matrix(transA, transB, m, n, k, alpha, beta, bgemm_a, bgemm_b, expect_c);
    cal_leading_dimension(transA, transB, &lda, &ldb);

    BlasExtendParam extend_param = {.type = BLAS_EXTEND_TYPE_PREPACK, .extra = prepack_mask, .next = NULL};

    if (prepack_mask & BLAS_EXTEND_PREPACK_A_MASK) {
        size_t size_a = kutacc_core_bgemm_pack_get_size('A', m, n, k);
        if (size_a != MATRIX_ROW * MATRIX_COL) {
            return false;
        }
        sa = (__bf16 *)malloc(size_a * sizeof(__bf16));

        kutacc_core_bgemm_pack('A', transA, transB, m, n, k, lda, ldb, bgemm_a, sa);
        kutacc_core_bgemm_ex(transA, transB, m, n, k, alpha, sa, lda, bgemm_b, ldb, beta, bgemm_c, ldc, &extend_param);
    } else if (prepack_mask & BLAS_EXTEND_PREPACK_B_MASK) {
        size_t size_b = kutacc_core_bgemm_pack_get_size('B', m, n, k);
        if (size_b != MATRIX_ROW * MATRIX_COL) {
            return false;
        }
        sb = (__bf16 *)malloc(size_b * sizeof(__bf16));

        kutacc_core_bgemm_pack('B', transA, transB, m, n, k, lda, ldb, bgemm_b, sb);
        kutacc_core_bgemm_ex(transA, transB, m, n, k, alpha, bgemm_a, lda, sb, ldb, beta, bgemm_c, ldc, &extend_param);
    } else {
        printf("input param prepack_mask error!\n");
        return false;
    }

    // compare bgemm_ex result
    for (int i = 0; i < MATRIX_ROW * MATRIX_ROW; i++) {
        if (fabs(vcvtah_f32_bf16(bgemm_c[i]) - vcvtah_f32_bf16(expect_c[i])) > BF16_EPS) {
            return false;
        }
    }

    if (sa) {
        free(sa);
    }
    if (sb) {
        free(sb);
    }

    return true;
}

void test_kutacc_bgemm_ex()
{
    tests_count++;
    printf("start test test_kutacc.kutacc_bgemm_ex_001\n");
    bool result = test_bgemm_ex('N', 'N', BLAS_EXTEND_PREPACK_A_MASK);
    if (result == true) {
        passed_count++;
        printf("end test test_kutacc.kutacc_bgemm_ex_001 PASSED\n");
    } else {
        failed_count++;
        printf("end test test_kutacc.kutacc_bgemm_ex_001 FAILED\n");
    }
}

int main()
{
    test_kutacc_bgemm_ex();
    printf("%u tests ran, %u tests PASSED, %u test FAILED.\n", tests_count, passed_count, failed_count);

    return 0;
}
