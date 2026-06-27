#include "kml_fft_test.h"
#include <stdlib.h>

#include "kfft.h"

int kml_fft_c2c_01()
{
    int rank = 2;
    int *n = (int*)kml_fft_malloc(sizeof(int) * rank);
    n[0] = 2;
    n[1] = 3;

    double init[6][2] = {{120, 0}, {8, 8}, {0, 0}, {0, 16}, {0, 16}, {-8, 8}};
    kml_fft_complex *in;
    in = (kml_fft_complex*)kml_fft_malloc(sizeof(kml_fft_complex) * n[0] * n[1]);
    for (int i = 0; i < n[0] * n[1]; i++) {
        in[i][0] = init[i][0];
        in[i][1] = init[i][1];
    }

    kml_fft_complex *out;
    out = (kml_fft_complex*)kml_fft_malloc(sizeof(kml_fft_complex) * n[0] * n[1]);

    kml_fft_plan plan;
    plan = kml_fft_plan_dft(rank, n, in, out, KML_FFT_FORWARD, KML_FFT_ESTIMATE);
    if (!plan) {
        return 0;
    }

    kml_fft_execute_dft(plan, in, out);

    kml_fft_destroy_plan(plan);
    kml_fft_free(n);
    kml_fft_free(in);
    kml_fft_free(out);

    return 1; // 成功
}