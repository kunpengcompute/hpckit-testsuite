#include "kml_vml_test.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "kvml.h"


int kml_vml_sin_01()
{
    int i, len = 4; 
    float src[] = {0.0f, 2.0f, INFINITY, NAN}; 
    float* dst = (float*)malloc(sizeof(float) * len); 
    if (dst == NULL) { 
        printf("Malloc Failed!\n"); 
        return 0; 
    } 
    vssin(len, src, dst); 

    return 1; // 成功
}