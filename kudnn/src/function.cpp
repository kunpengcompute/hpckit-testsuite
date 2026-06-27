#include "function.hpp"

using SizeType = KuDNN::SizeType;
using Shape = KuDNN::Shape;
using TensorInfo = const KuDNN::TensorInfo;
KuDNN::Element::TypeT TypeF16 = KuDNN::Element::TypeT::F16;
KuDNN::Element::TypeT TypeF32 = KuDNN::Element::TypeT::F32;
const int MAX_THS = 100;
const float FACTOR_THS = 0.02;
float errBound = 1e-3;
float maxError = 1e-3;

template <typename T>
struct typeMap;

template <>
struct typeMap<__fp16> {
    static constexpr KuDNN::Element::TypeT val = KuDNN::Element::TypeT::F16;
};

template <>
struct typeMap<float> {
    static constexpr KuDNN::Element::TypeT val = KuDNN::Element::TypeT::F32;
};

template <typename T>
static void conv2dRef(Shape &srcShape, Shape &weiShape, Shape &dstShape, Shape &strides, Shape &dilates,
                      Shape &paddingL, Shape &paddingR, const T *src, const T *wei, T *dst, const T *bia)
{
    SizeType N = srcShape[0], IC = srcShape[1], IH = srcShape[2], IW = srcShape[3];
    SizeType KH = weiShape[2], KW = weiShape[3];
    SizeType OC = dstShape[1], OH = dstShape[2], OW = dstShape[3];
    SizeType SH = strides[0], SW = strides[1];
    SizeType DH = dilates[0], DW = dilates[1];
    SizeType PH_L = paddingL[0], PW_L = paddingL[1];
    SizeType PH_R = paddingR[0], PW_R = paddingR[1];
    OH = (IH + PH_L + PH_R - 1 - (KH - 1) * (DH + 1)) / SH + 1;
    OW = (IW + PW_L + PW_R - 1 - (KW - 1) * (DW + 1)) / SW + 1;
    int threadsUsed = (int)ceil((float)(N * OC * OH * OW) * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
#pragma omp parallel for collapse(4) num_threads(threadsUsed)
    for (SizeType n = 0; n < N; ++n) {
        for (SizeType oc = 0; oc < OC; ++oc) {
            T biaVal = *(bia + oc);
            for (SizeType oh = 0; oh < OH; ++oh) {
                for (SizeType ow = 0; ow < OW; ++ow) {
                    T sum = (T)0.0f;
                    for (SizeType ic = 0; ic < IC; ++ic) {
                        for (SizeType kh = 0; kh < KH; ++kh) {
                            for (SizeType kw = 0; kw < KW; ++kw) {
                                SizeType srcX = oh * SH + kh * (DH + 1) - PH_L;
                                SizeType srcY = ow * SW + kw * (DW + 1) - PW_L;
                                T srcVal = (T)0.0f;
                                if (srcX >= 0 && srcX < IH && srcY >= 0 && srcY < IW) {
                                    srcVal = *(src + n * IC * IH * IW + ic * IH * IW + srcX * IW + srcY);
                                }
                                T weiVal = *(wei + oc * IC * KH * KW + ic * KH * KW + kh * KW + kw);
                                sum += srcVal * weiVal;
                            }
                        }
                    }
                    *(dst + n * OC * OH * OW + oc * OH * OW + oh * OW + ow) = sum + biaVal;
                }
            }
        }
    }
}

template <typename T>
static bool Conv2dFWDFunc1(Shape &srcShape, Shape &weiShape, Shape &dstShape, Shape &strides, Shape &dilates,
                           Shape &paddingL, Shape &paddingR, KuDNN::ConvolutionAlgorithm &alg)
{
    SizeType N = srcShape[0], IC = srcShape[1], IH = srcShape[2], IW = srcShape[3];
    SizeType OC = dstShape[1], OH = dstShape[2], OW = dstShape[3];
    SizeType KH = weiShape[2], KW = weiShape[3];
    const KuDNN::TensorInfo srcTensor = {srcShape, typeMap<T>::val, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightsTensor = {weiShape, typeMap<T>::val, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {dstShape, typeMap<T>::val, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{OC}, typeMap<T>::val, KuDNN::Layout::A};

    KuDNN::ConvolutionLayerFWD convFwdLayer1(srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates, paddingL,
                                            paddingR, alg);

    SizeType srcSize = N * IC * IH * IW;
    SizeType dstSize = N * OC * OH * OW;
    SizeType weiSize = OC * IC * KH * KW;
    SizeType biaSize = OC;
    T *src = (T *)malloc(srcSize * sizeof(T));
    T *dst = (T *)malloc(dstSize * sizeof(T));
    T *dstRef = (T *)malloc(dstSize * sizeof(T));
    T *wei = (T *)malloc(weiSize * sizeof(T));
    T *bia = (T *)malloc(biaSize * sizeof(T));
    if (src == nullptr || dst == nullptr || dstRef == nullptr || wei == nullptr || bia == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return false;
    }
    // generate random test data
    std::uniform_real_distribution<float> u(-1, 1);
    std::default_random_engine e(time(NULL));
    int threadsUsed = (int)ceil((float)srcSize * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
    for (SizeType i = 0; i < srcSize; ++i) {
        *(src + i) = (T)u(e);
    }
    for (SizeType i = 0; i < weiSize; ++i) {
        *(wei + i) = (T)u(e);
    }
    for (SizeType i = 0; i < biaSize; ++i) {
        *(bia + i) = (T)u(e);
    }
    convFwdLayer1.Run(src, wei, dst, bia);
    conv2dRef<T>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, src, wei, dstRef, bia);
    float error = 0.0;
#pragma omp parallel for reduction(+ : error) num_threads(threadsUsed)
    for (SizeType i = 0; i < dstSize; ++i) {
        error += *(dst + i) - *(dstRef + i);
    }
    free(src);
    free(dst);
    free(dstRef);
    free(wei);
    free(bia);
    error = std::abs(error) / dstSize;
    
    return error < errBound;
}

bool kudnn_conv_01()
{
    SizeType N = 4, IC = 4, IH = 128, IW = 100;
    SizeType OC = 5, OH = 0, OW = 0;
    SizeType KH = 3, KW = 3;
    Shape strides(1, 1), dilates(1, 1), paddingL(1, 1), paddingR(1, 1);
    OH = (IH + paddingL[0] + paddingR[0] - 1 - (KH - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OW = (IW + paddingL[1] + paddingR[1] - 1 - (KW - 1) * (dilates[1] + 1)) / strides[1] + 1;
    Shape srcShape(N, IC, IH, IW), weiShape(OC, IC, KH, KW), dstShape(N, OC, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    return Conv2dFWDFunc1<float>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
}

template <typename T>
static void conv3dRef(Shape &srcShape, Shape &weiShape, Shape &dstShape, Shape &strides, Shape &dilates,
                      Shape &paddingL, Shape &paddingR, const T *src, const T *wei, T *dst, const T *bia)
{
    SizeType N = srcShape[0], IC = srcShape[1], ID = srcShape[2], IH = srcShape[3], IW = srcShape[4];
    SizeType KD = weiShape[2], KH = weiShape[3], KW = weiShape[4];
    SizeType OC = dstShape[1], OD = dstShape[2], OH = dstShape[2], OW = dstShape[3];
    SizeType SD = strides[0], SH = strides[1], SW = strides[2];
    SizeType DD = dilates[0], DH = dilates[1], DW = dilates[2];
    SizeType PD_L = paddingL[0], PH_L = paddingL[1], PW_L = paddingL[2];
    SizeType PD_R = paddingR[0], PH_R = paddingR[1], PW_R = paddingR[2];
    OD = (ID + PD_L + PD_R - 1 - (KD - 1) * (DD + 1)) / SD + 1;
    OH = (IH + PH_L + PH_R - 1 - (KH - 1) * (DH + 1)) / SH + 1;
    OW = (IW + PW_L + PW_R - 1 - (KW - 1) * (DW + 1)) / SW + 1;
    int threadsUsed = (int)ceil((float)(N * OC * OD * OH * OW) * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
#pragma omp parallel for collapse(5) num_threads(threadsUsed)
    for (SizeType n = 0; n < N; ++n) {
        for (SizeType oc = 0; oc < OC; ++oc) {
            T biaVal = *(bia + oc);
            for (SizeType od = 0; od < OD; ++od) {
                for (SizeType oh = 0; oh < OH; ++oh) {
                    for (SizeType ow = 0; ow < OW; ++ow) {
                        T sum = (T)0.0f;
                        for (SizeType ic = 0; ic < IC; ++ic) {
                            for (SizeType kd = 0; kd < KD; ++kd) {
                                for (SizeType kh = 0; kh < KH; ++kh) {
                                    for (SizeType kw = 0; kw < KW; ++kw) {
                                        SizeType srcZ = od * SD + kd * (DD + 1) - PD_L;
                                        SizeType srcX = oh * SH + kh * (DH + 1) - PH_L;
                                        SizeType srcY = ow * SW + kw * (DW + 1) - PW_L;
                                        T srcVal = (T)0.0f;
                                        if (srcZ >= 0 && srcZ < ID && srcX >= 0 && srcX < IH && srcY >= 0 &&
                                            srcY < IW) {
                                            srcVal = *(src + n * IC * ID * IH * IW + ic * ID * IH * IW +
                                                       srcZ * IH * IW + srcX * IW + srcY);
                                        }
                                        T weiVal = *(wei + oc * IC * KD * KH * KW + ic * KD * KH * KW + kd * KH * KW +
                                                     kh * KW + kw);
                                        sum += srcVal * weiVal;
                                    }
                                }
                            }
                        }
                        *(dst + n * OC * OD * OH * OW + oc * OD * OH * OW + od * OH * OW + oh * OW + ow) = sum + biaVal;
                    }
                }
            }
        }
    }
}

template <typename T>
static bool Conv3dFWDFunc1(Shape &srcShape, Shape &weiShape, Shape &dstShape, Shape &strides, Shape &dilates,
                           Shape &paddingL, Shape &paddingR, KuDNN::ConvolutionAlgorithm &alg)
{
    SizeType N = srcShape[0], IC = srcShape[1], ID = srcShape[2], IH = srcShape[3], IW = srcShape[4];
    SizeType OC = dstShape[1], OD = dstShape[2], OH = dstShape[3], OW = dstShape[4];
    SizeType KD = weiShape[2], KH = weiShape[3], KW = weiShape[4];
    const KuDNN::TensorInfo srcTensor = {srcShape, typeMap<T>::val, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightsTensor = {weiShape, typeMap<T>::val, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {dstShape, typeMap<T>::val, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{OC}, typeMap<T>::val, KuDNN::Layout::A};

    KuDNN::ConvolutionLayerFWD convFwdLayer1(srcTensor, weightsTensor, dstTensor, biasTensor, strides, dilates, paddingL,
                                            paddingR, alg);

    SizeType srcSize = N * IC * ID * IH * IW;
    SizeType dstSize = N * OC * OD * OH * OW;
    SizeType weiSize = OC * IC * KD * KH * KW;
    SizeType biaSize = OC;
    T *src = (T *)malloc(srcSize * sizeof(T));
    T *dst = (T *)malloc(dstSize * sizeof(T));
    T *dstRef = (T *)malloc(dstSize * sizeof(T));
    T *wei = (T *)malloc(weiSize * sizeof(T));
    T *bia = (T *)malloc(biaSize * sizeof(T));
    if (src == nullptr || dst == nullptr || dstRef == nullptr || wei == nullptr || bia == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return false;
    }
    // generate random test data
    std::uniform_real_distribution<float> u(-1, 1);
    std::default_random_engine e(time(NULL));
    int threadsUsed = (int)ceil((float)srcSize * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
    for (SizeType i = 0; i < srcSize; ++i) {
        *(src + i) = (T)u(e);
    }
    for (SizeType i = 0; i < weiSize; ++i) {
        *(wei + i) = (T)u(e);
    }
    for (SizeType i = 0; i < biaSize; ++i) {
        *(bia + i) = (T)u(e);
    }
    conv3dRef<T>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, src, wei, dstRef, bia);
    convFwdLayer1.Run(src, wei, dst, bia);
    float error = 0.0;
#pragma omp parallel for reduction(+ : error) num_threads(threadsUsed)
    for (SizeType i = 0; i < dstSize; ++i) {
        error += *(dst + i) - *(dstRef + i);
    }
    free(src);
    free(dst);
    free(dstRef);
    free(wei);
    free(bia);
    error = std::abs(error) / dstSize;
    
    return error < errBound;
}

bool kudnn_conv_02()
{
    SizeType N = 1, IC = 5, ID = 20, IH = 90, IW = 160;
    SizeType OC = 10, OD = 0, OH = 0, OW = 0;
    SizeType KD = 1, KH = 1, KW = 1;
    Shape strides(1, 1, 1), dilates(0, 0, 0), paddingL(0, 0, 0), paddingR(0, 0, 0);
    OD = (ID + paddingL[0] + paddingR[0] - 1 - (KD - 1) * (dilates[0] + 1)) / strides[0] + 1;
    OH = (IH + paddingL[1] + paddingR[1] - 1 - (KH - 1) * (dilates[1] + 1)) / strides[1] + 1;
    OW = (IW + paddingL[2] + paddingR[2] - 1 - (KW - 1) * (dilates[2] + 1)) / strides[2] + 1;
    Shape srcShape(N, IC, ID, IH, IW), weiShape(OC, IC, KD, KH, KW), dstShape(N, OC, OD, OH, OW);
    KuDNN::ConvolutionAlgorithm alg(KuDNN::ConvolutionAlgorithm::AUTO);
    return Conv3dFWDFunc1<__fp16>(srcShape, weiShape, dstShape, strides, dilates, paddingL, paddingR, alg);
}

template <typename T>
static void gnormRef(Shape shape, SizeType groupInfo, const T *src, T *dst, const T *scale, const T *shift,
                     bool scaleApply, bool shiftApply, float *mean, float *variance, const float eps,
                     bool global_stats = false)
{
    SizeType innerSize = 1;
    SizeType numDims = shape.GetNumDims();
    if (numDims == 4) {
        innerSize = shape[2] * shape[3];
    } else if (numDims == 5) {
        innerSize = shape[2] * shape[3] * shape[4];
    } else if (numDims == 3) {
        innerSize = shape[2];
    } else {
        return;
    }
    SizeType A = shape[0], B = shape[1];
    SizeType chPerGroup = B / groupInfo;
    innerSize *= chPerGroup;
    int threadsUsed = (int)ceil((float)(A * groupInfo) * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
    if (!global_stats) {
        // mean compute
#pragma omp parallel for collapse(2) num_threads(threadsUsed)
        for (SizeType a = 0; a < A; ++a) {
            for (SizeType g = 0; g < groupInfo; ++g) {
                float sum = 0.0f;
                for (SizeType i = 0; i < innerSize; ++i) {
                    sum += (float)*(src + a * groupInfo * innerSize + g * innerSize + i);
                }
                *(mean + a * groupInfo + g) = sum / innerSize;
            }
        }
        // variance compute
#pragma omp parallel for collapse(2) num_threads(threadsUsed)
        for (SizeType a = 0; a < A; ++a) {
            for (SizeType g = 0; g < groupInfo; ++g) {
                float sum = 0.0f;
                float meanVal = *(mean + a * groupInfo + g);
                for (SizeType i = 0; i < innerSize; ++i) {
                    float diff = (float)*(src + a * groupInfo * innerSize + g * innerSize + i) - meanVal;
                    sum += diff * diff;
                }
                *(variance + a * groupInfo + g) = sum / innerSize;
            }
        }
    }
    // norm
    innerSize /= chPerGroup;
#pragma omp parallel for collapse(2) num_threads(threadsUsed)
    for (SizeType a = 0; a < A; ++a) {
        for (SizeType g = 0; g < groupInfo; ++g) {
            float meanVal = *(mean + a * groupInfo + g);
            float invStd = 1.0f / std::sqrt(*(variance + a * groupInfo + g) + eps);
            for (SizeType gi = 0; gi < chPerGroup; ++gi) {
                int b = g * chPerGroup + gi;
                for (SizeType i = 0; i < innerSize; ++i) {
                    float srcVal = (float)*(src + a * B * innerSize + b * innerSize + i);
                    float normalized = (srcVal - meanVal) * invStd;
                    if (scaleApply) {
                        normalized *= scale[b];
                    }
                    if (shiftApply) {
                        normalized += shift[b];
                    }
                    *(dst + a * B * innerSize + b * innerSize + i) = static_cast<T>(normalized);
                }
            }
        }
    }
}
// forward
template <typename T>
static bool GnormForwardFunc1(const TensorInfo &srcInfo, const TensorInfo &scaleShiftInfo, SizeType groupInfo,
                              const TensorInfo &dstInfo, KuDNN::NormalizationFlags flags)
{
    KuDNN::GroupNormalizationLayerFWD gnormLayer1(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags);

    SizeType srcSize = srcInfo.GetTotalTensorSize();
    SizeType dstSize = dstInfo.GetTotalTensorSize();
    SizeType statSize = srcInfo.GetDims()[0] * groupInfo;
    SizeType scaleSize = scaleShiftInfo.GetTotalTensorSize();

    T *src = (T *)malloc(srcSize * sizeof(T));
    T *dst = (T *)malloc(srcSize * sizeof(T));
    T *dstRef = (T *)malloc(srcSize * sizeof(T));
    float *mean = (float *)malloc(statSize * sizeof(float));
    float *variance = (float *)malloc(statSize * sizeof(float));
    T *scale = (T *)malloc(scaleSize * sizeof(T));
    T *shift = (T *)malloc(scaleSize * sizeof(T));
    float eps = 1e-5;
    if (src == nullptr || dst == nullptr || dstRef == nullptr || mean == nullptr || variance == nullptr ||
        scale == nullptr || shift == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return false;
    }

    bool global_stats = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_GLOBAL_STATS);

    // generate random test data
    std::uniform_real_distribution<float> u(-1, 1);
    std::default_random_engine e(time(NULL));
    int threadsUsed = (int)ceil((float)srcSize * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
    for (SizeType i = 0; i < srcSize; ++i) {
        *(src + i) = (T)u(e);
    }
    for (SizeType i = 0; i < scaleSize; ++i) {
        *(scale + i) = (T)u(e);
        *(shift + i) = (T)u(e);
    }
    if (global_stats) {
        for (SizeType i = 0; i < statSize; ++i) {
            *(mean + i) = 1.0f;
            *(variance + i) = 0.1f;
        }
    }

    gnormLayer1.Run(src, dst, scale, shift, mean, variance, true, eps);
    bool scaleApply = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_SCALE);
    bool shiftApply = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_SHIFT);

    gnormRef<T>(srcInfo.GetDims(), groupInfo, src, dstRef, scale, shift, scaleApply, shiftApply, mean, variance, eps,
                global_stats);
    float error = 0.0;
#pragma omp parallel for reduction(+ : error) num_threads(threadsUsed)
    for (SizeType i = 0; i < dstSize; ++i) {
        error += *(dst + i) - *(dstRef + i);
    }
    free(src);
    free(dst);
    free(dstRef);
    free(mean);
    free(variance);
    free(scale);
    free(shift);
    error = std::abs(error) / dstSize;
    
    return error < errBound;
}

bool kudnn_gnorm_01()
{
    Shape shape(4, 128, 18, 320);
    SizeType groupInfo = 2;
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    TensorInfo scaleShiftInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::ABCD};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    return GnormForwardFunc1<__fp16>(srcInfo, scaleShiftInfo, groupInfo, dstInfo, flags);
}

template <typename T>
static void lnormRef(SizeType outerSize, SizeType innerSize, const T *src, T *dst, const T *scale, const T *shift,
                     bool scaleApply, bool shiftApply, float *mean, float *variance, const float eps,
                     bool global_stats = false)
{
    int threadsUsed = (int)ceil((float)outerSize * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
    if (!global_stats) {
        // mean compute
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
        for (SizeType i = 0; i < outerSize; ++i) {
            float sum = 0.0f;
            for (SizeType j = 0; j < innerSize; ++j) {
                sum += (float)*(src + i * innerSize + j);
            }
            *(mean + i) = sum / innerSize;
        }
        // variance compute
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
        for (SizeType i = 0; i < outerSize; ++i) {
            float sum = 0.0f;
            float meanVal = *(mean + i);
            for (SizeType j = 0; j < innerSize; ++j) {
                float diff = (float)*(src + i * innerSize + j) - meanVal;
                sum += diff * diff;
            }
            *(variance + i) = sum / innerSize;
        }
    }
    // norm
#pragma omp parallel for collapse(2) num_threads(threadsUsed)
    for (SizeType i = 0; i < outerSize; ++i) {
        float meanVal = *(mean + i);
        float varVal = *(variance + i);
        float invStd = 1.0f / std::sqrt(varVal + eps);
        for (SizeType j = 0; j < innerSize; ++j) {
            float srcVal = (float)*(src + i * innerSize + j);
            float normalized = (srcVal - meanVal) * invStd;
            if (scaleApply) {
                normalized *= scale[j];
            }
            if (shiftApply) {
                normalized += shift[j];
            }
            *(dst + i * innerSize + j) = static_cast<T>(normalized);
        }
    }
}
// forward
template <typename T>
static bool LnormForwardFunc1(const TensorInfo &srcInfo, const TensorInfo &statInfo, const TensorInfo &scaleShiftInfo,
                              const TensorInfo &dstInfo, KuDNN::NormalizationFlags flags)
{
    KuDNN::NormalizationLayerFWD lnormLayer1(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags);
    SizeType srcSize = srcInfo.GetTotalTensorSize();
    SizeType dstSize = dstInfo.GetTotalTensorSize();
    SizeType statSize = statInfo.GetTotalTensorSize();
    SizeType innerSize = scaleShiftInfo.GetTotalTensorSize();
    T *src = (T *)malloc(srcSize * sizeof(T));
    T *dst = (T *)malloc(dstSize * sizeof(T));
    T *dstRef = (T *)malloc(dstSize * sizeof(T));
    float *mean = (float *)malloc(statSize * sizeof(float));
    float *variance = (float *)malloc(statSize * sizeof(float));
    T *scale = (T *)malloc(innerSize * sizeof(T));
    T *shift = (T *)malloc(innerSize * sizeof(T));
    float eps = 1e-5;
    if (src == nullptr || dst == nullptr || dstRef == nullptr || mean == nullptr || variance == nullptr ||
        scale == nullptr || shift == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return false;
    }

    bool global_stats = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_GLOBAL_STATS);

    // generate random test data
    std::uniform_real_distribution<float> u(-1, 1);
    std::default_random_engine e(time(NULL));
    int threadsUsed = (int)ceil((float)srcSize * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
    for (SizeType i = 0; i < srcSize; ++i) {
        *(src + i) = (T)u(e);
    }
    for (SizeType j = 0; j < innerSize; ++j) {
        *(scale + j) = (T)u(e);
        *(shift + j) = (T)u(e);
    }
    if (global_stats) {
        for (SizeType i = 0; i < statSize; i++) {
            *(mean + i) = 1.0f;
            *(variance + i) = 0.1f;
        }
    }

    float error = 0.0f;
    lnormLayer1.Run(src, dst, scale, shift, mean, variance, true, eps);
    bool scaleApply = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_SCALE);
    bool shiftApply = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_SHIFT);
    lnormRef<T>(statSize, innerSize, src, dstRef, scale, shift, scaleApply, shiftApply, mean, variance, eps,
                global_stats);
#pragma omp parallel for reduction(+ : error) num_threads(threadsUsed)
    for (SizeType i = 0; i < dstSize; ++i) {
        error += *(dst + i) - *(dstRef + i);
    }
    free(src);
    free(dst);
    free(dstRef);
    free(mean);
    free(variance);
    free(scale);
    free(shift);
    error = std::abs(error) / dstSize;
    
    return error < errBound;
}

bool kudnn_lnorm_01()
{
    Shape shape(2, 3240, 1152);
    TensorInfo srcInfo = {shape, TypeF32, KuDNN::Layout::ABC};
    TensorInfo statInfo = {{shape[0], shape[1]}, TypeF32, KuDNN::Layout::AB};
    TensorInfo scaleShiftInfo = {{shape[2]}, TypeF32, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF32, KuDNN::Layout::ABC};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::NONE;
    return LnormForwardFunc1<float>(srcInfo, statInfo, scaleShiftInfo, dstInfo, flags);
}

template <typename T>
static void siluSimple(T *dst, SizeType size)
{
    if (!dst || size == 0)
        return;
    for (SizeType i = 0; i < size; ++i) {
        T sigmoid_x = static_cast<T>(1) / (static_cast<T>(1) + std::exp(static_cast<float>(-dst[i])));
        dst[i] = dst[i] * sigmoid_x;
    }
}

template <typename T>
static void linearActivationSimple(SizeType batch, SizeType m, SizeType n, SizeType k, const T *src, T *weight, T *dst,
                                   T *bias, KuDNN::ActivationFunction kind)
{
    for (SizeType bh = 0; bh < batch; ++bh) {
        for (SizeType i = 0; i < m; ++i) {
            for (SizeType j = 0; j < n; ++j) {
                T sum = 0;
                for (SizeType l = 0; l < k; ++l) {
                    sum += src[bh * m * k + i * k + l] * weight[bh * n * k + n * l + j];
                }
                dst[bh * m * n + i * n + j] = sum + bias[bh * m * n + i * n + j];
            }
        }
    }
    if (kind == KuDNN::ActivationFunction::SWISH) {
        siluSimple<T>(dst, batch * m * n);
    }
}

template <typename T>
static void floatDataInit(T **src, T **weight, T **dst, T **dstRef, T **bias, SizeType srcTotalSize,
                          SizeType weightTotalSize, SizeType dstTotalSize)
{
    *src = (T *)malloc(srcTotalSize * sizeof(T));
    *weight = (T *)malloc(weightTotalSize * sizeof(T));
    *dst = (T *)malloc(dstTotalSize * sizeof(T));
    *dstRef = (T *)malloc(dstTotalSize * sizeof(T));
    *bias = (T *)malloc(dstTotalSize * sizeof(T));

    if (*src == nullptr || *weight == nullptr || *dst == nullptr || *dstRef == nullptr || *bias == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return;
    }
    std::uniform_real_distribution<float> u(-1, 1);
    static std::default_random_engine e(time(NULL));

    for (SizeType i = 0; i < srcTotalSize; ++i) {
        (*src)[i] = (T)u(e);
    }
    for (SizeType i = 0; i < weightTotalSize; ++i) {
        (*weight)[i] = (T)u(e);
    }
    for (SizeType i = 0; i < dstTotalSize; ++i) {
        (*bias)[i] = (T)u(e);
    }
}

static std::pair<float, float> compareEle(float ref, float diff, KuDNN::Element::TypeT type, int K)
{
    float e = 0.0f;
    float eps = 0.0f;
    std::pair<float, float> pairEps = {0.0f, 0.0f};
    switch (type) {
        case KuDNN::Element::TypeT::F16:
            eps = 1e-3 * K;
            e = (std::fabs(ref) > eps) ? diff / ref : static_cast<float>(diff);
            pairEps = {e, eps};
            break;
        case KuDNN::Element::TypeT::BF16:
            eps = 1e-2 * K;
            e = (std::fabs(ref) > eps) ? diff / ref : static_cast<float>(diff);
            pairEps = {e, eps};
            break;
        case KuDNN::Element::TypeT::F32:
            eps = 1e-4;
            e = (std::fabs(ref) > 1e-4) ? diff / ref : static_cast<float>(diff);
            pairEps = {e, eps};
            break;
        default:
            eps = 1;
            e = diff;
            pairEps = {e, eps};
            break;
    }
    return pairEps;
}

template <typename T>
static float findMaxValue(T *value1, SizeType valueSize1, T *value2, SizeType valueSize2)
{
    T maxVal = value1[0];

    for (SizeType i = 0; i < valueSize1; ++i) {
        maxVal = std::max(maxVal, value1[i]);
    }

    for (SizeType i = 0; i < valueSize2; ++i) {
        maxVal = std::max(maxVal, value2[i]);
    }
    return static_cast<float>(maxVal);
}

template <typename T>
static bool Test_Template_Function(const KuDNN::TensorInfo srcTensor, const KuDNN::TensorInfo weightTensor,
                                   const KuDNN::TensorInfo dstTensor, const KuDNN::TensorInfo biasTensor, float alpha,
                                   float beta, KuDNN::ActivationFunction algKind, int numThreads)
{
    KuDNN::LinearActivationLayerFWD linearActivationLayerFwd(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta,
                                                            algKind, numThreads);

    T *src = nullptr, *weight = nullptr, *dst = nullptr, *dstRef = nullptr, *bias = nullptr;
    floatDataInit<T>(&src, &weight, &dst, &dstRef, &bias, srcTensor.GetTotalTensorSize(),
                     weightTensor.GetTotalTensorSize(), dstTensor.GetTotalTensorSize());

    Shape srcShape = srcTensor.GetDims();
    Shape weightShape = weightTensor.GetDims();
    SizeType srcDimNums = srcShape.GetNumDims();
    SizeType weightDimNums = weightShape.GetNumDims();
    SizeType m = srcShape[srcDimNums - 2];
    SizeType n = weightShape[weightDimNums - 1];
    SizeType k = srcShape[srcDimNums - 1];
    SizeType batch = dstTensor.GetTotalTensorSize() / (m * n);

    linearActivationSimple(batch, m, n, k, src, weight, dstRef, bias, algKind);

    linearActivationLayerFwd.Run(src, weight, dst, bias, numThreads);

    float maxValue = findMaxValue<T>(src, batch * m * k, weight, batch * n * k);

    bool flag = true;
    for (SizeType i = 0; i < batch * m * n; ++i) {
        float diff = std::abs(static_cast<float>(*(dst + i) - *(dstRef + i)));
        auto pairEps = compareEle(maxValue, diff, KuDNN::Element::MatchType<T>(), k);

        if (std::abs(pairEps.first) > pairEps.second) {
            flag = false;
            break;
        }
    }
    if (src != nullptr) {
        free(src);
    }
    if (weight != nullptr) {
        free(weight);
    }
    if (dst != nullptr) {
        free(dst);
    }
    if (dstRef != nullptr) {
        free(dstRef);
    }
    if (bias != nullptr) {
        free(bias);
    }
    return flag;
}
bool kudnn_linearActivation_01()
{
    const KuDNN::TensorInfo srcTensor = {{3, 2, 110, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo weightTensor = {{3, 2, 20, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo dstTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    const KuDNN::TensorInfo biasTensor = {{3, 2, 110, 200}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCD};
    float alpha = 1.0f;
    float beta = 0.0f;
    KuDNN::ActivationFunction algKind = KuDNN::ActivationFunction::SWISH;
    int numThreads = 0;
    return Test_Template_Function<float>(srcTensor, weightTensor, dstTensor, biasTensor, alpha, beta, algKind, numThreads);
}

static void linearSimpleFloat(SizeType batch, SizeType m, SizeType n, SizeType k, const float *src, float *weight,
                              float *dst, float *bias, int transSrc = 0, int transDst = 0)
{
    for (SizeType bh = 0; bh < batch; ++bh) {
        for (SizeType i = 0; i < m; ++i) {
            for (SizeType j = 0; j < n; ++j) {
                float sum = 0;
                for (SizeType l = 0; l < k; ++l) {
                    if (transSrc == 0 && transDst == 0) {
                        sum += src[bh * m * k + i * k + l] * weight[bh * n * k + n * l + j];
                    } else if (transSrc == 1 && transDst == 0) {
                        sum += src[bh * m * k + i * k + l] * weight[bh * n * k + j * k + l];
                    } else if (transSrc == 0 && transDst == 1) {
                        sum += src[bh * m * k + l * m + i] * weight[bh * n * k + n * l + j];
                    } else {
                        sum += src[bh * m * k + l * m + i] * weight[bh * n * k + j * k + l];
                    }
                }
                if (bias) {
                    dst[bh * m * n + i * n + j] = sum + bias[bh * m * n + i * n + j];
                } else {
                    dst[bh * m * n + i * n + j] = sum;
                }
            }
        }
    }
}
template <typename T>
static void DataInit(T **src, T **weight, T **dst, T **dstRef, T **bias, T **res, T **linearRes, T **linearResRef,
                     SizeType srcTotalSize, SizeType weightTotalSize, SizeType dstTotalSize, SizeType resTotalSize,
                     SizeType linearResTotalSize)
{
    *src = (T *)malloc(srcTotalSize * sizeof(T));
    *weight = (T *)malloc(weightTotalSize * sizeof(T));
    *dst = (T *)malloc(dstTotalSize * sizeof(T));
    *dstRef = (T *)malloc(dstTotalSize * sizeof(T));
    *bias = (T *)malloc(dstTotalSize * sizeof(T));
    *res = (T *)malloc(resTotalSize * sizeof(T));
    *linearRes = (T *)malloc(linearResTotalSize * sizeof(T));
    *linearResRef = (T *)malloc(linearResTotalSize * sizeof(T));

    if (*src == nullptr || *weight == nullptr || *dst == nullptr || *dstRef == nullptr || *bias == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return;
    }
    std::uniform_real_distribution<float> u(-1, 1);
    static std::default_random_engine e(time(NULL));

    for (SizeType i = 0; i < srcTotalSize; ++i) {
        (*src)[i] = (T)u(e);
    }
    for (SizeType i = 0; i < weightTotalSize; ++i) {
        (*weight)[i] = (T)u(e);
    }
    for (SizeType i = 0; i < dstTotalSize; ++i) {
        (*bias)[i] = (T)u(e);
    }

    for (SizeType i = 0; i < resTotalSize; ++i) {
        (*res)[i] = (T)u(e);
    }
}

static void addFloat(const float *dst, float *res, float *linearRes, float gamma, SizeType size)
{
    for (SizeType i = 0; i < size; i++) {
        linearRes[i] = dst[i] + res[i] * gamma;
    }
}

static std::pair<float, float> LresCompareEle(float ref, float diff, KuDNN::Element::TypeT type, SizeType K, SizeType L)
{
    float e = 0.0f;
    float eps = 0.0f;
    std::pair<float, float> pairEps = {0.0f, 0.0f};
    switch (type) {
        case KuDNN::Element::TypeT::F16:
            eps = 1e-3 * K * L;
            e = (std::fabs(ref) > eps) ? diff / ref : static_cast<float>(diff);
            pairEps = {e, eps};
            break;
        case KuDNN::Element::TypeT::BF16:
            eps = 1e-2 * K * L;
            e = (std::fabs(ref) > eps) ? diff / ref : static_cast<float>(diff);
            pairEps = {e, eps};
            break;
        case KuDNN::Element::TypeT::F32:
            eps = 1e-4;
            e = (std::fabs(ref) > 1e-4) ? diff / ref : static_cast<float>(diff);
            pairEps = {e, eps};
            break;
        default:
            eps = 1;
            e = diff;
            pairEps = {e, eps};
            break;
    }
    return pairEps;
}

static bool Test_Template_Function_FP32(const KuDNN::TensorInfo srcTensor, const KuDNN::TensorInfo weightTensor,
                                        const KuDNN::TensorInfo dstTensor, const KuDNN::TensorInfo biasTensor,
                                        const KuDNN::TensorInfo resTensor, const KuDNN::TensorInfo linearResTensor,
                                        float alpha, float beta, float gamma, KuDNN::ResOpsFunction algKind,
                                        int numThreads, int transSrc = 0, int transDst = 0)
{
    KuDNN::LinearResFWD LinearResFWD(srcTensor, weightTensor, dstTensor, biasTensor, resTensor, linearResTensor, alpha,
                                    beta, gamma, algKind, numThreads);

    float *src = nullptr, *weight = nullptr, *dst = nullptr, *dstRef = nullptr, *bias = nullptr, *res = nullptr,
          *linearRes = nullptr, *linearResRef = nullptr;
    DataInit<float>(&src, &weight, &dst, &dstRef, &bias, &res, &linearRes, &linearResRef,
                    srcTensor.GetTotalTensorSize(), weightTensor.GetTotalTensorSize(), dstTensor.GetTotalTensorSize(),
                    resTensor.GetTotalTensorSize(), linearResTensor.GetTotalTensorSize());

    Shape srcShape = srcTensor.GetDims();
    Shape weightShape = weightTensor.GetDims();
    Shape resShape = resTensor.GetDims();
    SizeType srcDimNums = srcShape.GetNumDims();
    SizeType weightDimNums = weightShape.GetNumDims();
    SizeType resDimNums = resShape.GetNumDims();
    SizeType dst_row = srcShape[srcDimNums - 2];        // m
    SizeType dst_col = weightShape[weightDimNums - 1];  // n
    SizeType src_col = srcShape[srcDimNums - 1];        // k
    SizeType res_col = resShape[resDimNums - 1];        // l
    SizeType LinearBatch = dstTensor.GetTotalTensorSize() / (dst_row * dst_col);
    SizeType PostOpsBatch = linearResTensor.GetTotalTensorSize() / (dst_row * res_col);

    SizeType L = 1;  // compareEle: K *L

    linearSimpleFloat(LinearBatch, dst_row, dst_col, src_col, src, weight, dstRef, bias, transSrc, transDst);
    if (algKind == KuDNN::ResOpsFunction::RES_MUL) {
        L = dst_col;
        linearSimpleFloat(PostOpsBatch, dst_row, res_col, dst_col, dstRef, res, linearResRef, nullptr, transSrc,
                          transDst);
    } else if (algKind == KuDNN::ResOpsFunction::RES_IDENTIAL) {
        addFloat(dstRef, res, linearResRef, 1.0f, linearResTensor.GetTotalTensorSize());
    } else {
        addFloat(dstRef, res, linearResRef, gamma, linearResTensor.GetTotalTensorSize());
    }

    LinearResFWD.Run(src, weight, dst, bias, res, linearRes, gamma);

    float maxValue =
        findMaxValue<float>(src, srcTensor.GetTotalTensorSize(), weight, weightTensor.GetTotalTensorSize());

    bool flag = true;
    for (SizeType i = 0; i < linearResTensor.GetTotalTensorSize(); ++i) {
        float diff = std::abs(static_cast<float>(*(linearRes + i) - *(linearResRef + i)));
        auto pairEps = LresCompareEle(maxValue, diff, KuDNN::Element::MatchType<float>(), src_col, L);
        if (std::abs(pairEps.first) > pairEps.second) {
            flag = false;
            break;
        }
    }

    if (src != nullptr) {
        free(src);
    }
    if (weight != nullptr) {
        free(weight);
    }
    if (dst != nullptr) {
        free(dst);
    }
    if (dstRef != nullptr) {
        free(dstRef);
    }
    if (bias != nullptr) {
        free(bias);
    }
    if (res != nullptr) {
        free(res);
    }
    if (linearRes != nullptr) {
        free(linearRes);
    }
    if (linearResRef != nullptr) {
        free(linearResRef);
    }
    return flag;
}

bool kudnn_linearRes_01()
{
    const KuDNN::TensorInfo srcTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo weightTensor = {{4, 3, 2, 20, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo dstTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo biasTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo resTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    const KuDNN::TensorInfo linearResTensor = {{4, 3, 2, 30, 20}, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABCDE};
    float alpha = 1.0f;
    float beta = 0.0f;
    float gamma = 2.0f;
    KuDNN::ResOpsFunction algKind = KuDNN::ResOpsFunction::RES_IDENTIAL;
    int numThreads = 0;
    return Test_Template_Function_FP32(srcTensor, weightTensor, dstTensor, biasTensor, resTensor, linearResTensor, alpha, beta,
                                gamma, algKind, numThreads);
}

template <typename T>
static void rmsnormRef(SizeType outerSize, SizeType innerSize, const T *src, T *dst, const T *scale, bool scaleApply,
                       float *variance, const float eps, bool global_stats = false)
{
    int threadsUsed = (int)ceil((float)outerSize * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
    // variance compute
    if (!global_stats) {
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
        for (SizeType i = 0; i < outerSize; ++i) {
            float sum = 0.0f;
            for (SizeType j = 0; j < innerSize; ++j) {
                float srcVal = (float)*(src + i * innerSize + j);
                sum += srcVal * srcVal;
            }
            *(variance + i) = sum / innerSize;
        }
    }
    // norm
#pragma omp parallel for collapse(2) num_threads(threadsUsed) schedule(static)
    for (SizeType i = 0; i < outerSize; ++i) {
        float varVal = *(variance + i);
        float invStd = 1.0f / std::sqrt(varVal + eps);
        for (SizeType j = 0; j < innerSize; ++j) {
            float srcVal = (float)*(src + i * innerSize + j);
            float normalized = srcVal * invStd;
            if (scaleApply) {
                normalized *= scale[j];
            }
            *(dst + i * innerSize + j) = static_cast<T>(normalized);
        }
    }
}
// forward
template <typename T>
static bool RnormForwardFunc1(const TensorInfo &srcInfo, const TensorInfo &statInfo, const TensorInfo &scaleInfo,
                              const TensorInfo &dstInfo, KuDNN::NormalizationFlags flags)
{
    KuDNN::RMSNormalizationLayerFWD rmsLayer1(srcInfo, statInfo, scaleInfo, dstInfo, flags);

    SizeType srcSize = srcInfo.GetTotalTensorSize();
    SizeType dstSize = dstInfo.GetTotalTensorSize();
    SizeType statSize = statInfo.GetTotalTensorSize();
    SizeType innerSize = scaleInfo.GetTotalTensorSize();

    T *src = (T *)malloc(srcSize * sizeof(T));
    T *dst = (T *)malloc(dstSize * sizeof(T));
    T *dstRef = (T *)malloc(dstSize * sizeof(T));
    float *variance = (float *)malloc(statSize * sizeof(float));
    T *scale = (T *)malloc(innerSize * sizeof(T));
    float eps = 1e-5;
    if (src == nullptr || dst == nullptr || dstRef == nullptr || variance == nullptr || scale == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return false;
    }

    bool global_stats = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_GLOBAL_STATS);

    // generate random test data
    std::uniform_real_distribution<float> u(-1, 1);
    std::default_random_engine e(time(NULL));
    int threadsUsed = (int)ceil((float)srcSize * FACTOR_THS);
    threadsUsed = threadsUsed > MAX_THS ? MAX_THS : threadsUsed;
#pragma omp parallel for num_threads(threadsUsed) schedule(static)
    for (SizeType i = 0; i < srcSize; ++i) {
        *(src + i) = (T)u(e);
    }
    for (SizeType j = 0; j < innerSize; ++j) {
        *(scale + j) = (T)u(e);
    }
    if (global_stats) {
        for (SizeType i = 0; i < statSize; i++) {
            *(variance + i) = 0.1f;
        }
    }

    float error = 0.0;
    rmsLayer1.Run(src, dst, scale, variance, true, eps);
    bool scaleApply = static_cast<bool>(flags & KuDNN::NormalizationFlags::USE_SCALE);
    rmsnormRef<T>(statSize, innerSize, src, dstRef, scale, scaleApply, variance, eps, global_stats);
#pragma omp parallel for reduction(+ : error) num_threads(threadsUsed)
    for (SizeType i = 0; i < dstSize; ++i) {
        error += *(dst + i) - *(dstRef + i);
    }
    free(src);
    free(dst);
    free(dstRef);
    free(variance);
    free(scale);
    error = std::abs(error) / dstSize;
    
    return error < errBound;
}

bool kudnn_rnorm_01()
{
    Shape shape(100, 100);
    TensorInfo srcInfo = {shape, TypeF16, KuDNN::Layout::AB};
    TensorInfo statInfo = {{shape[0]}, TypeF16, KuDNN::Layout::A};
    TensorInfo scaleInfo = {{shape[1]}, TypeF16, KuDNN::Layout::A};
    TensorInfo dstInfo = {shape, TypeF16, KuDNN::Layout::AB};
    KuDNN::NormalizationFlags flags = KuDNN::NormalizationFlags::USE_SHIFT;
    return RnormForwardFunc1<__fp16>(srcInfo, statInfo, scaleInfo, dstInfo, flags);
}

void softmaxHelp(float *src, int start, int end)
{
    if (start > end || !src)
        return;

    float max_val = src[start];
    for (int i = start + 1; i <= end; ++i) {
        if (src[i] > max_val) {
            max_val = src[i];
        }
    }
    float sum_exp = 0.0f;
    for (int i = start; i <= end; ++i) {
        sum_exp += std::exp(src[i] - max_val);
    }
    for (int i = start; i <= end; ++i) {
        src[i] = std::exp(src[i] - max_val) / sum_exp;
    }
}

static void softmaxSimple(Shape shape, float *src)
{
    SizeType numDims = shape.GetNumDims();
    SizeType outSize = 1;
    for (SizeType i = 0; i < numDims - 1; ++i) {
        outSize *= shape[i];
    }
    SizeType innerSize = shape[numDims - 1];

    for (SizeType i = 0; i < outSize; ++i) {
        softmaxHelp(src, i * innerSize, (i + 1) * innerSize - 1);
    }
}

static void dataInit(float **src, float **dst, float **dstRef, SizeType totalSize)
{
    *src = (float *)malloc(totalSize * sizeof(float));
    *dst = (float *)malloc(totalSize * sizeof(float));
    *dstRef = (float *)malloc(totalSize * sizeof(float));

    if (*src == nullptr || *dst == nullptr || *dstRef == nullptr) {
        std::cerr << "Memory allocation failed" << std::endl;
        return;
    }
    std::uniform_real_distribution<float> u(-1, 1);
    std::default_random_engine e(time(NULL));

    for (SizeType i = 0; i < totalSize; ++i) {
        (*src)[i] = u(e);
    }
    memcpy(*dstRef, *src, totalSize * sizeof(float));
}

static void dataFree(float **src, float **dst, float **dstRef)
{
    free(*src);
    free(*dst);
    free(*dstRef);

    *src = nullptr;
    *dst = nullptr;
    *dstRef = nullptr;
}
static bool FWD_3D_FP32(const Shape &shape)
{
    const KuDNN::TensorInfo srcTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    const KuDNN::TensorInfo dstTensor = {shape, KuDNN::Element::TypeT::F32, KuDNN::Layout::ABC};
    SizeType axis = 2;
    KuDNN::SoftmaxLayerFWD softmaxLayerFwd(srcTensor, dstTensor, axis, KuDNN::SoftmaxAlgorithmKind::SOFTMAX);

    float *src = nullptr, *dst = nullptr, *dstRef = nullptr;
    SizeType totalSize = srcTensor.GetTotalTensorSize();
    dataInit(&src, &dst, &dstRef, totalSize);
    softmaxSimple(shape, dstRef);
    softmaxLayerFwd.Run(src, dst);
    for (SizeType i = 0; i < totalSize; ++i) {
        float error = std::abs(*(dst + i) - *(dstRef + i));
        if (error > maxError) {
            return false;
        }
    }
    dataFree(&src, &dst, &dstRef);
    return true;
}
bool kudnn_softmax_01()
{
    Shape shape(20, 40, 50);
    return FWD_3D_FP32(shape);
}