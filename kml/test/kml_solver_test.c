#include "kml_solver_test.h"
#include <stdlib.h>
#include <stdio.h>
#include "kml_solver.h"
void PrepareMatrix(KmlSolverMatrixStore *store,
                    KmlSolverMatrixOption *opt, int nRow, int nCol)
{
    store->indexType = KMLSS_INDEX_INT32;
    store->valueType = KMLSS_VALUE_FP64;
    store->nRow = nRow;
    store->nCol = nCol;
    opt->fieldMask = KMLSS_MATRIX_OPTION_TYPE;
    opt->type = KMLSS_MATRIX_GEN;
}

int kml_solver_tydss_01()
{
int ierr;
    int n = 8;
    int nrhs = 1;

    // Create matrix A
    int ia[9] = {0, 2, 4, 6, 7, 8, 10, 12, 14};
    int ja[14] = {0, 7, 1, 6, 2, 5, 3, 4, 2, 5, 1, 6, 0, 7};
    double a[14] = {1.0, 2.0, -2.0, 3.0, 3.0, 4.0, -4.0, 5.0, 4.0, -6.0, 3.0, 7.0, 2.0, 8.0};
    KmlSolverMatrixStore storeA;
    KmlSolverMatrixOption optA;
    PrepareMatrix(&storeA, &optA, n, n);
    storeA.format = KMLSS_MATRIX_STORE_CSR;
    storeA.csr.rowOffset = ia;
    storeA.csr.colIndex = ja;
    storeA.csr.value = a;

    KmlSolverMatrix *A;
    ierr = KmlSolverMatrixCreate(&A, &storeA, &optA);
    if (ierr != KMLSS_NO_ERROR) {
        printf("ERROR when create A: %d\n", ierr);
        return 1;
    }

    // Create vector b
    double b[8] = {3.0, 1.0, 7.0, -4.0, 5.0, -2.0, 10.0, 10.0};
    KmlSolverMatrixStore storeB;
    KmlSolverMatrixOption optB;
    PrepareMatrix(&storeB, &optB, n, nrhs);
    storeB.format = KMLSS_MATRIX_STORE_DENSE_COL_MAJOR;
    storeB.dense.value = b;
    storeB.dense.ld = n;
    KmlSolverMatrix *B;
    ierr = KmlSolverMatrixCreate(&B, &storeB, &optB);

    if (ierr != KMLSS_NO_ERROR) {
        printf("ERROR when create b: %d\n", ierr);
        return 1;
    }

    // Create vector x
    double x[8] = {0};
    KmlSolverMatrixStore storeX;
    KmlSolverMatrixOption optX;
    PrepareMatrix(&storeX, &optX, n, nrhs);
    storeX.format = KMLSS_MATRIX_STORE_DENSE_COL_MAJOR;
    storeX.dense.value = x;
    storeX.dense.ld = n;
    KmlSolverMatrix *X;
    ierr = KmlSolverMatrixCreate(&X, &storeX, &optX);
    if (ierr != KMLSS_NO_ERROR) {
        printf("ERROR when create x: %d\n", ierr);
        return 1;
    }

    // Init solver
    KmlDssInitOption opt;
    opt.fieldMask = KMLDSS_INIT_OPTION_NTHREADS;
    opt.nThreads = 32;
    KmlDssSolver *solver;
    ierr = KmlDssInit(&solver, &opt);
    if (ierr != KMLSS_NO_ERROR) {
        printf("ERROR in KmlDssInit: %d\n", ierr);
        return ierr;
    }

    // Analyze
    KmlDssAnalyzeOption optAnalyze;
    optAnalyze.fieldMask = KMLDSS_ANALYZE_OPTION_NTHREADS_RDR;
    optAnalyze.nThreadsRdr = 8;
    ierr = KmlDssAnalyze(solver, A, &optAnalyze);
    if (ierr != KMLSS_NO_ERROR) {
        printf("ERROR in KmlDssAnalyze: %d\n", ierr);
        return ierr;
    }

    // Factorize
    KmlDssFactorizeOption optFact;
    optFact.fieldMask = KMLDSS_FACTORIZE_OPTION_PERTURBATION_THRESHOLD;
    optFact.perturbationThreshold = 1e-8;
    ierr = KmlDssFactorize(solver, A, &optFact);
    if (ierr != KMLSS_NO_ERROR) {
        printf("ERROR in KmlDssFactorize: %d\n", ierr);
        return ierr;
    }

    // Solve
    KmlDssSolveOption optSolve;
    optSolve.fieldMask = KMLDSS_SOLVE_OPTION_SOLVE_STAGE |
                        KMLDSS_SOLVE_OPTION_REFINE_METHOD;
    optSolve.stage = KMLDSS_SOLVE_ALL;
    optSolve.refineMethod = KMLDSS_REFINE_OFF;
    ierr = KmlDssSolve(solver, B, X, &optSolve);
    if (ierr != KMLSS_NO_ERROR) {
        printf("ERROR in KmlDssSolve: %d\n", ierr);
        return ierr;
    }

    // Set new values of A
    double a1[14] = {2.0, 3.0, -3.0, 4.0, 4.0, 5.0, -5.0,
                        6.0, 5.0, -7.0, 4.0, 8.0, 3.0, 9.0};
    KmlSolverMatrixSetValue(A, a1);
    // Factorize with new values
    ierr = KmlDssFactorize(solver, A, &optFact);
    if (ierr != KMLSS_NO_ERROR) {
        printf("ERROR in KmlDssFactorize: %d\n", ierr);
        return ierr;
    }



    // Set new values of B
    double b1[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    KmlSolverMatrixSetValue(B, b1);
    // Solve with new values
    ierr = KmlDssSolve(solver, B, X, &optSolve);
    if (ierr != KMLSS_NO_ERROR) {
        printf("ERROR in KmlDssSolve: %d\n", ierr);
        return ierr;
    }

    // Query
    KmlDssInfo info;
    info.fieldMask = KMLDSS_INFO_PEAK_MEM;
    
    ierr = KmlDssQuery(solver, &info);
    if (ierr != KMLSS_NO_ERROR) {
        printf("ERROR in KmlDssQuery: %d\n", ierr);
        return ierr;
    }
    
    // Destroy
    KmlDssClean(&solver);
    KmlSolverMatrixDestroy(&A);
    KmlSolverMatrixDestroy(&B);
    KmlSolverMatrixDestroy(&X);
    return 0;
}