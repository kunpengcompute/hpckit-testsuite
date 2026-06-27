#include <gtest/gtest.h>
extern "C" {
#include "kml_fft_test.h"
#include "kml_blas_test.h"
#include "kml_lapack_test.h"
#include "kml_solver_test.h"
#include "kml_scalapack_test.h"
#include "kml_ipl_test.h"
#include "kml_conv_test.h"
#include "kml_vml_test.h"
#include "kml_vsl_test.h"
#include "kml_svml_test.h"
#include "kml_math_test.h"
#include "kml_spblas_test.h"
}
#include <iostream>
#include <fstream>

class CustomTestEventListener : public::testing::EmptyTestEventListener {
public:
    CustomTestEventListener() {
    }

    ~CustomTestEventListener() override {
    }

    void OnTestStart(const::testing::TestInfo& test_info) override {
        std::cout << "start test " <<test_info.name() << std::endl;
    }

    void OnTestEnd(const::testing::TestInfo& test_info) override {
        std::cout << "end test " <<test_info.name() << " PASSED" << std::endl;
    }

    void OnTestProgramEnd(const::testing::UnitTest& unit_test) override {
        int failed = unit_test.failed_test_count();
        int total = unit_test.total_test_count();
        std::cout << total << " tests ran, " << total - failed << " tests PASSED, "
        << failed << " tests FAILED." << std::endl;
    }
};

TEST(kml_test, kml_fft) {
    int result = kml_fft_c2c_01();
    EXPECT_EQ(result, 1) << "kml_fft_c2c_01 should succeed";
}

TEST(kml_test, kml_blas) {
    int result = kml_blas_sgemm_01();
    EXPECT_EQ(result, 1) << "kml_blas_sgemm_01 should succeed";
}

TEST(kml_test, kml_lapack) {
    int result = kml_lapack_dgetrf_01();
    EXPECT_EQ(result, 0) << "kml_lapack_dgetrf_01 should succeed";
}

TEST(kml_test, kml_solver) {
    int result = kml_solver_tydss_01();
    EXPECT_EQ(result, 0) << "kml_solver_tydss_01 should succeed";
}

TEST(kml_test, kml_scalapack) {
    int result = kml_scalapack_pdgetrf_01();
    EXPECT_EQ(result, 0) << "kml_scalapack_pdgetrf_01 should succeed";
}

TEST(kml_test, kml_ipl) {
    int result = kml_ipl_Cssmh_01();
    EXPECT_EQ(result, 1) << "kml_ipl_Cssmh_01 should succeed";
}

#ifdef IS_NEED_CONV
TEST(kml_test, kml_conv) {
    int result = kml_conv_conv2d_01();
    EXPECT_EQ(result, 1) << "kml_conv_conv2d_01 should succeed";
}
#endif

TEST(kml_test, kml_vml) {
    int result = kml_vml_sin_01();
    EXPECT_EQ(result, 1) << "kml_vml_sin_01 should succeed";
}

TEST(kml_test, kml_vsl) {
    int result = kml_vsl_uniform_01();
    EXPECT_EQ(result, 1) << "kml_vsl_uniform_01 should succeed";
}

TEST(kml_test, kml_svml) {
    int result = kml_svml_sin_01();
    EXPECT_EQ(result, 1) << "kml_svml_sin_01 should succeed";
}

TEST(kml_test, kml_math) {
    int result = kml_math_sin_01();
    EXPECT_EQ(result, 1) << "kml_math_sin_01 should succeed";
}

TEST(kml_test, kml_spblas) {
    int result = kml_spblas_spgemm_01();
    EXPECT_EQ(result, 1) << "kml_spblas_spgemm_01 should succeed";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new CustomTestEventListener());

    return RUN_ALL_TESTS();
}