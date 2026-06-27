#include "kml_ipl_test.h"
#include <stdlib.h>
#include <stdio.h>
#include "kipl.h"


int kml_ipl_Cssmh_01()
{
    int data_n = 11;
    float data_x[data_n];
    float data_y[data_n];
    for (int i = 0; i < data_n; i++) {
        data_x[i] = (float)(i) / (data_n - 1);
        data_y[i] = sin(15.0 * data_x[i]);
    }
    int test_n = 2 * data_n - 1;
    float test_x[test_n];
    float test_y[test_n];
    for (int i = 0; i < test_n; i++) {
        test_x[i] = (float)(i) / (test_n - 1);
        test_y[i] = sin(15.0 * test_x[i]);
    }
    float b[data_n], cscoef[4 * data_n];
    float weight[data_n];
    float smpar = data_n;
    for (int i = 0; i < data_n; i++) {
        weight[i] = 0.1;
    }
    KIPL_Cssmh(data_n, data_x, data_y, weight, smpar, cscoef);
    int nintv = data_n - 1;
    for (int i = 0; i < test_n; i++) {
        float y;
        float x = test_x[i];
        y = KIPL_Csval(x, nintv, data_x, cscoef);
    }
    return 1; // 成功
}