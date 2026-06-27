#include <omp.h>
#include <mpi.h>
#include <cstdio>
#include "kuqcd.h"

using namespace kuqcd;

template<typename T,int Nbatch, KuQCDReconstructType ReconS,KuQCDReconstructType ReconN>
bool TestCGMPC(const KuQCDGaugeOrder gauge_order);

int main(int argc, char** argv)
{
    int provided = 0;
    int required = MPI_THREAD_FUNNELED;
    int flag = MPI_Init_thread(&argc, &argv, required, &provided);
    bool is_pass = TestCGMPC<float, 1, KUQCD_GAUGE_RECON_NO, KUQCD_GAUGE_RECON_NO>(KUQCD_NORMAL_ORDER);
    if (is_pass) {
        printf("1 tests ran, 1 tests PASSED\n");   
    } else {
        printf("1 tests ran, 1 tests FAILED\n");  
    }

    MPI_Finalize();
    return 0;
}

template<typename T,int Nbatch, KuQCDReconstructType ReconS,KuQCDReconstructType ReconN>
bool TestCGMPC(const KuQCDGaugeOrder gauge_order)
{
    printf("start test kuqcd_solve_multishift\n");
    int repos = 1;
    int nx = 4, ny = 4, nz = 4, nt = 1, seed = 1;
    const int num_shift = 1;
    std::array<unsigned short, 4> gridDim{1, 1, 1, 1};
    KuQCDRankOrder od = KUQCD_TZYX_ORDER;
    KuQCDSiteSubset site_subset = KUQCD_ODD;
    int haloDepthSpin = 0;
    int haloDepthGauge = 3;

    Lattice latSup(nx ,ny ,nz ,nt);

    KuQCDPrecision prec = KUQCD_SINGLE;
    LattBase * latBaseOdd = nullptr;
    KuQCDGaugeParam Gagugeoptions;
    Gagugeoptions.lat[0] = nx;
    Gagugeoptions.lat[1] = ny;
    Gagugeoptions.lat[2] = nz;
    Gagugeoptions.lat[3] = nt;
    Gagugeoptions.gauge_order = gauge_order;
    Gagugeoptions.od = od;
    Gagugeoptions.prec = prec;
    site_subset = KUQCD_ALL;
    Gagugeoptions.site_subset = site_subset;
    Gagugeoptions.halo_depth = haloDepthGauge;
    
    double shifts[num_shift];
    shifts[0] = 3.671676313381e-05;
    FILE* fp;
    float* origin;
    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int rk;
    MPI_Comm_rank(MPI_COMM_WORLD, &rk);

    size_t vol = (size_t) nx*ny*nz*nt /Nbatch/np;
    size_t volh = vol / 2;
    size_t spinorLen = volh * Nbatch * 6;
    size_t gaugeLen = vol * Nbatch * 4 * ReconS;
    size_t gaugeLenNaik = vol * Nbatch * 4 * ReconN;

    origin = (float *)malloc(((num_shift + 1) * spinorLen + gaugeLen + gaugeLenNaik) * sizeof(float));
    char name[100];

    for (int i = 0 ;i<(num_shift + 1)*spinorLen + gaugeLen + gaugeLenNaik;i++)origin[i] = 1.0;
    kuqcd_init();
    float *originSrc = origin;
    float *origingaugaeSmeared = &(origin[(num_shift + 1)*spinorLen]);
    float * origingaugaeNaik = &(origin[(num_shift + 1)*spinorLen + gaugeLen]);

    Gagugeoptions.link_type = KUQCD_FAT_LINK;
    Gagugeoptions.recon = ReconS;
    kuqcd_load_gaugefield((void*)origingaugaeSmeared, &Gagugeoptions);
    Gagugeoptions.link_type = KUQCD_LONG_LINK;
    Gagugeoptions.recon = ReconN;
    kuqcd_load_gaugefield((void*)origingaugaeNaik, &Gagugeoptions);

    KuQCDInvertParam invertOpts;
    invertOpts.dslash_type = KUQCD_HISQPC_DIRAC;
    invertOpts.inverter_type = KUQCD_CGM;
    invertOpts.matpc_type = KUQCD_ODD_ODD;
    int iter = 0;
    invertOpts.num_shift = num_shift;
    invertOpts.max_iter = 6500;
    invertOpts.tol = 1e-4;
    invertOpts.shifts = shifts;
    invertOpts.prec = prec;
    void *dst[num_shift] = {nullptr};
    float *dst_value[num_shift];
    float *dst_Ref[num_shift];
    for(int i = 0;i<num_shift;i++) {
        dst_value[i] = (float*)malloc(spinorLen*sizeof(float));
        dst[i] = (void*)(dst_value[i]);
        dst_Ref[i] = (float*)malloc(spinorLen*sizeof(float));
        for (int j = 0;j < spinorLen ;j++) {
            dst_Ref[i][j] = 0.007561;
        }
    }
    double tStart = MPI_Wtime();

    kuqcd_solve_multishift(dst, (void*)originSrc, &invertOpts);
    double tEnd = MPI_Wtime();
    fflush(0);
    printf("total time =  %e [s]\n", tEnd - tStart);
    fflush(stdout);
    double diff = 0;
    for (int i = 0 ;i < num_shift ; i++) {
        for (int j = 0; j < spinorLen ; j++) {
            diff += abs(dst_value[i][j] - dst_Ref[i][j]);
        }
    }
    diff /= spinorLen;
    bool is_pass = true;
    if (diff < invertOpts.tol) {
        printf("end test kuqcd_solve_multishift PASSED\n");
    } else {
        printf("end test kuqcd_solve_multishift FAILED\n");
        is_pass = false;
    }
    return is_pass;
}
