#include "kml_conv_test.h"
#include <stdlib.h>
#include <stdio.h>
#include "conv.h"


int kml_conv_conv2d_01()
{
    int batch = 1;
    int inputChannels = 1;
    int inputHeight = 6;
    int inputWidth = 6;
    int kernelHeight = 3;
    int kernelWidth = 3;
    int strideY = 1;
    int strideX = 1;
    int padHeight = 0;
    int padWidth = 0;
    int dilationY = 1;
    int dilationX = 1;
    int outputChannels = 1;
    float input[36] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                       7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                       13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
                       19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                       25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
                       31.0, 32.0, 33.0, 34.0, 35.0, 36.0};
    float kernel[9] = {1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0,
                       7.0, 8.0, 9.0};

    float *bias = NULL;
    int outputHeight = (inputHeight + 2 * padHeight - dilationY * (kernelHeight - 1) - 1) / strideY + 1;
    int outputWidth = (inputWidth + 2 * padWidth - dilationX * (kernelWidth - 1) - 1) / strideX + 1;

    float output[16] = {0.0};
    conv2d_fp32(input, batch, inputChannels, inputHeight, inputWidth, kernel, kernelHeight, kernelWidth, strideY, strideX, padHeight, padWidth, dilationY, dilationX, bias, output, outputChannels);

    return 1; // 成功
}