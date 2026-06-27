#include "kml_spblas_test.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "kspblas.h"


int kml_spblas_spgemm_01()
{
    kml_sparse_operation_t opt = KML_SPARSE_OPERATION_NON_TRANSPOSE; 
    KML_INT m = 3; 
    KML_INT n = 3; 
    KML_INT k = 3; 
    float alpha = 1.0; 
    float beta = 1.0; 
    char *matdescra = "G00C"; //一般矩阵，基0索引 
    float val[4] = {9, 8, 5, 2}; 
    KML_INT indx[4] = {2, 1, 2, 2}; 
    KML_INT pntrb[3] = {0, 2, 3}; 
    KML_INT pntre[3] = {2, 3, 4}; 
    float b[9] = {1, 5, 7, 4, 7, 7, 3, 3, 7}; 
    float c[9] = {0, 8, 8, 0, 2, 8, 3, 5, 6}; 
    KML_INT ldb = 3; 
    KML_INT ldc = 3; 
    kml_sparse_status_t status = kml_sparse_scsrmm(opt, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b, ldb, beta, c, ldc); 


    return 1; // 成功
}