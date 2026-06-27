#include <iostream>
#include <cmath>
#include "kulitho_image.h"
static uint32_t tests_count = 0;
static uint32_t passed_count = 0;
static uint32_t failed_count = 0;
double real_out_upsampling[256] = {
    -0.010007, 0.005362, 0.034187, 0.062500, 0.082022, 0.093750, 0.105478, 0.125000, 0.153313, 0.182138, 0.197507, 0.187500, 0.149672, 0.093750, 0.037828, -0.010007,
	0.005362, 0.034187, 0.062500, 0.082022, 0.093750, 0.105478, 0.125000, 0.153313, 0.182138, 0.197507, 0.187500, 0.149672, 0.093750, 0.037828, -0.010007, 0.005362,
	0.034187, 0.062500, 0.082022, 0.093750, 0.105478, 0.125000, 0.153313, 0.182138, 0.197507, 0.187500, 0.149672, 0.093750, 0.037828, -0.010007, 0.005362, 0.034187,
	0.062500, 0.082022, 0.093750, 0.105478, 0.125000, 0.153313, 0.182138, 0.197507, 0.187500, 0.149672, 0.093750, 0.037828, -0.010007, 0.005362, 0.034187, 0.062500,
	0.082022, 0.093750, 0.105478, 0.125000, 0.153313, 0.182138, 0.197507, 0.187500, 0.149672, 0.093750, 0.037828, -0.010007, 0.005362, 0.034187, 0.062500, 0.082022,
	0.093750, 0.105478, 0.125000, 0.153313, 0.182138, 0.197507, 0.187500, 0.149672, 0.093750, 0.037828, -0.010007, 0.005362, 0.034187, 0.062500, 0.082022, 0.093750,
	0.105478, 0.125000, 0.153313, 0.182138, 0.197507, 0.187500, 0.149672, 0.093750, 0.037828, -0.010007, 0.005362, 0.034187, 0.062500, 0.082022, 0.093750, 0.105478,
	0.125000, 0.153313, 0.182138, 0.197507, 0.187500, 0.149672, 0.093750, 0.037828, -0.010007, 0.005362, 0.034187, 0.062500, 0.082022, 0.093750, 0.105478, 0.125000,
	0.153313, 0.182138, 0.197507, 0.187500, 0.149672, 0.093750, 0.037828, -0.010007, 0.005362, 0.034187, 0.062500, 0.082022, 0.093750, 0.105478, 0.125000, 0.153313,
	0.182138, 0.197507, 0.187500, 0.149672, 0.093750, 0.037828, -0.010007, 0.005362, 0.034187, 0.062500, 0.082022, 0.093750, 0.105478, 0.125000, 0.153313, 0.182138,
	0.197507, 0.187500, 0.149672, 0.093750, 0.037828, -0.010007, 0.005362, 0.034187, 0.062500, 0.082022, 0.093750, 0.105478, 0.125000, 0.153313, 0.182138, 0.197507,
	0.187500, 0.149672, 0.093750, 0.037828, -0.010007, 0.005362, 0.034187, 0.062500, 0.082022, 0.093750, 0.105478, 0.125000, 0.153313, 0.182138, 0.197507, 0.187500,
	0.149672, 0.093750, 0.037828, -0.010007, 0.005362, 0.034187, 0.062500, 0.082022, 0.093750, 0.105478, 0.125000, 0.153313, 0.182138, 0.197507, 0.187500, 0.149672,
	0.093750, 0.037828, -0.010007, 0.005362, 0.034187, 0.062500, 0.082022, 0.093750, 0.105478, 0.125000, 0.153313, 0.182138, 0.197507, 0.187500, 0.149672, 0.093750,
	0.037828, -0.010007, 0.005362, 0.034187, 0.062500, 0.082022, 0.093750, 0.105478, 0.125000, 0.153313, 0.182138, 0.197507, 0.187500, 0.149672, 0.093750, 0.037828
};
double real_out_lapacian[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

int verify_upsampling(double *out, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (abs(out[i * cols + j] - real_out_upsampling[i * cols + j]) > 1e-5)
                // printf("%f, %f\n", out[i * cols + j], real_out_upsampling[i * cols + j]);
                return 0;
        }
    }
    return 1;
}
int verify_laplacian(double *out, size_t rows, size_t cols)
{
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (abs(out[i * cols + j] - real_out_lapacian[i * cols + j]) > 1e-5)
                // printf("%f, %f\n", out[i * cols + j], real_out_lapacian[i * cols + j]);
                return 0;
        }
    }
    return 1;
}

int kulitho_upsampling_01()
{
    tests_count++;    
    printf("start test kulitho_upsampling_01\n");
    size_t src_rows = 4, src_cols = 4;
    size_t dst_rows = src_rows * 4;
    size_t dst_cols = src_cols * 4;

    double *grid_in = new double[src_rows * src_cols];
    for (int i = 0; i < src_rows; ++i) {
        for (int j = 0; j < src_cols; ++j) {
            grid_in[i * src_cols + j] = j;
        }
    }
    double *grid_out = new double[dst_rows * dst_cols];
    kulitho::kulitho_FFTupsampling(grid_in, grid_out, src_rows, src_cols, dst_rows, dst_cols, 0, 0);

    if (verify_upsampling(grid_out, dst_rows, dst_cols)) {
        passed_count++;
        std::cout << "end test kulitho_upsampling_01: PASSED" << std::endl;
    } else {
        failed_count++;
        std::cout << "end test kulitho_upsampling_01: FAILED" << std::endl;
    }

    delete[] grid_in;
    delete[] grid_out;
    return 0;
}

int kulitho_laplacian_01()
{
    tests_count++;
    printf("start test kulitho_laplacian_01\n");
    size_t rows = 3, cols = 3;
    double src[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double dst[rows * cols];

    kulitho::kulitho_laplacian(src, rows, cols, dst);
    if (verify_laplacian(dst, rows, cols)) {
        passed_count++;
        std::cout << "end test kulitho_laplacian_01: PASSED" << std::endl;
    } else {
        failed_count++;
        std::cout << "end test kulitho_laplacian_01: FAILED" << std::endl;
    }
    return 0;
}

int main()
{
    kulitho_upsampling_01();
    kulitho_laplacian_01();
    printf("%u tests ran, %u tests PASSED, %u test FAILED.\n", tests_count, passed_count, failed_count);
}