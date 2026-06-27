#include "kml_scalapack_test.h"

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "kscalapack.h"
void blacs_get_(int*, int*, int*);
void blacs_gridinit_(int*, char*, int*, int*);
void blacs_gridinfo_(int*, int*, int*, int*, int*);
void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
void blacs_gridexit_(int*);
int numroc_(int*, int*, int*, int*, int*);

int kml_scalapack_pdgetrf_01() {
    // 默认值
    int izero=0;
    int ione=1;
    int N = 1000; // 矩阵维度
    int MB = 64, NB = 64; // 块大小
    int P = 1, Q = 1; // 进程网格尺寸

    // 初始化MPI环境
    MPI_Init(NULL, NULL);
    int iam, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &iam);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    if (P * Q != np) {
        if (iam == 0) {
            fprintf(stderr, "Error: The product of P and Q must equal the number of processes.\n");
        }
        MPI_Finalize();
        return -1;
    }

    // 创建BLACS上下文
    int ictxt, myrow, mycol;
    blacs_get_(&izero, &izero, &ictxt); // -> Create context
    blacs_gridinit_(&ictxt, "Row", &P, &Q);
    blacs_gridinfo_(&ictxt, &P, &Q, &myrow, &mycol);

    // 定义矩阵分布描述符
    int desca[9], descb[9];
    int info;

    // 计算本地矩阵尺寸
    int M_loc = numroc_(&N, &MB, &myrow, &izero, &P);
    int N_loc = numroc_(&N, &NB, &mycol, &izero, &Q);

    // 分配局部矩阵存储空间
    double *A = (double *)malloc(M_loc * N_loc * sizeof(double));
    if (!A) {
        printf("Process %d could not allocate memory for A\n", iam);
        return -1;
    }

    // 初始化局部矩阵数据
    for (int i = 0; i < M_loc * N_loc; ++i) {
        A[i] = drand48(); // 使用随机数初始化
    }

    // 设置矩阵描述符
    descinit_(desca, &N, &N, &MB, &NB, &izero, &izero, &ictxt, &M_loc, &info);


    // 调用ScaLAPACK pdgetrf函数进行LU分解
    int *ipiv = (int *)malloc((M_loc + MB) * sizeof(int));
    pdgetrf_(&N, &N, A, &ione, &ione, desca, ipiv, &info);

    // 清理资源
    free(A);
    free(ipiv);
    blacs_gridexit_(&ictxt);
    MPI_Finalize();

    return info;
}