#include "kml_svml_test.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ksvml.h"


int kml_svml_sin_01()
{
    float32x4_t src = {0.0f, 2.0f, INFINITY, NAN};
    float32x4_t dst = svml128_sin_f32(src);
    return 1; // 成功
}