#include "kml_lapack_test.h"
#include "klapack.h"
#include <stdlib.h>

int kml_lapack_dgetrf_01()
{
    char trans = 'N';
    int n = 5;
    int nrhs = 1;
    int lda = 5;
    int ldb = 5;
    int ipiv[5];
    int info = 0;
    double a[]={72.1673 , 66.1857 , 64.7644 , 28.0199 , 91.4151,
                6.5180 , 62.8483 , 72.4323 , 46.5760 ,  8.6928,
                28.9821 , 42.1828 , 18.6437 , 99.8612 , 35.6972,
                67.9812 ,  5.0880 , 85.5035 ,79.2945 , 54.5920,
                28.6869 , 49.7512 ,  7.5186 ,28.6929 , 84.6041};
    double b[]={1.0, 2.0, 3.0, 4.0, 5.0};
    dgetrf_(&n, &n, a, &lda, ipiv, &info);
    dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
    return info;
}