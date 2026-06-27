#include "kml_vsl_test.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "krng.h"


int kml_vsl_uniform_01()
{
    VSLStreamStatePtr stream;
    unsigned seed = 42;
    int errcode = vslNewStream(&stream, VSL_BRNG_MCG59, seed);
    if (errcode != VSL_STATUS_OK) {
        fprintf(stderr, "Failure in newstream\n");
        return 0;
    }

    SizeType n = 10;
    float *r = (float *)malloc(sizeof(float) * n);
    if (r == NULL) {
        fprintf(stderr, "Failure in malloc\n");
        return 0;
    }

    if (vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, r, 0.0, 1.0)) {
        fprintf(stderr, "Failure in vsRngUniform\n");
        goto out;
    }

    /* deinitialize the stream */
    errcode = vslDeleteStream(&stream);
    if (errcode != VSL_STATUS_OK) {
        fprintf(stderr, "Failure in deleting stream\n");
        goto out;
    }

out:
    free(r);

    return 1; // 成功
}