#include "kudnn.hpp"
#include <vector>
#include <string>
#include <string.h>
#include <unordered_map>
#include <random>
#include <iostream>

static std::unordered_map<std::string, KuDNN::Layout> layoutMap = {
    {"KuDNN::Layout::UNDEFINED", KuDNN::Layout::UNDEFINED},
    {"KuDNN::Layout::A", KuDNN::Layout::A},
    {"KuDNN::Layout::AB", KuDNN::Layout::AB},
    {"KuDNN::Layout::BA", KuDNN::Layout::BA},
    {"KuDNN::Layout::ABC", KuDNN::Layout::ABC},
    {"KuDNN::Layout::ACB", KuDNN::Layout::ACB},
    {"KuDNN::Layout::BAC", KuDNN::Layout::BAC},
    {"KuDNN::Layout::BCA", KuDNN::Layout::BCA},
    {"KuDNN::Layout::CAB", KuDNN::Layout::CAB},
    {"KuDNN::Layout::CBA", KuDNN::Layout::CBA},
    {"KuDNN::Layout::ABCD", KuDNN::Layout::ABCD},
    {"KuDNN::Layout::ABDC", KuDNN::Layout::ABDC},
    {"KuDNN::Layout::ACBD", KuDNN::Layout::ACBD},
    {"KuDNN::Layout::ACDB", KuDNN::Layout::ACDB},
    {"KuDNN::Layout::ADBC", KuDNN::Layout::ADBC},
    {"KuDNN::Layout::ADCB", KuDNN::Layout::ADCB},
    {"KuDNN::Layout::BACD", KuDNN::Layout::BACD},
    {"KuDNN::Layout::BCDA", KuDNN::Layout::BCDA},
    {"KuDNN::Layout::CDAB", KuDNN::Layout::CDAB},
    {"KuDNN::Layout::CDBA", KuDNN::Layout::CDBA},
    {"KuDNN::Layout::DCAB", KuDNN::Layout::DCAB},
    {"KuDNN::Layout::ABCDE", KuDNN::Layout::ABCDE},
    {"KuDNN::Layout::ABCED", KuDNN::Layout::ABCED},
    {"KuDNN::Layout::ABDEC", KuDNN::Layout::ABDEC},
    {"KuDNN::Layout::ACBDE", KuDNN::Layout::ACBDE},
    {"KuDNN::Layout::ACDEB", KuDNN::Layout::ACDEB},
    {"KuDNN::Layout::ADECB", KuDNN::Layout::ADECB},
    {"KuDNN::Layout::BACDE", KuDNN::Layout::BACDE},
    {"KuDNN::Layout::BCDEA", KuDNN::Layout::BCDEA},
    {"KuDNN::Layout::CDEAB", KuDNN::Layout::CDEAB},
    {"KuDNN::Layout::CDEBA", KuDNN::Layout::CDEBA},
    {"KuDNN::Layout::DECAB", KuDNN::Layout::DECAB},
    {"KuDNN::Layout::ROW_MAJOR", KuDNN::Layout::ROW_MAJOR},
    {"KuDNN::Layout::COL_MAJOR", KuDNN::Layout::COL_MAJOR},
    {"KuDNN::Layout::NCHW", KuDNN::Layout::NCHW},
    {"KuDNN::Layout::NHWC", KuDNN::Layout::NHWC},
    {"KuDNN::Layout::NCDHW", KuDNN::Layout::NCDHW},
    {"KuDNN::Layout::NDHWC", KuDNN::Layout::NDHWC},
    {"KuDNN::Layout::OIHW", KuDNN::Layout::OIHW},
    {"KuDNN::Layout::HWIO", KuDNN::Layout::HWIO},
    {"KuDNN::Layout::HWOI", KuDNN::Layout::HWOI},
    {"KuDNN::Layout::OHWI", KuDNN::Layout::OHWI},
    {"KuDNN::Layout::OHWO", KuDNN::Layout::OHWO},
    {"KuDNN::Layout::IOHW", KuDNN::Layout::IOHW},
};

static std::unordered_map<std::string, KuDNN::Element::TypeT> typeTMap = {
    {"UNDEFINED", KuDNN::Element::TypeT::UNDEFINED},
    {"KuDNN::Element::TypeT::F32", KuDNN::Element::TypeT::F32},
    {"KuDNN::Element::TypeT::F16", KuDNN::Element::TypeT::F16},
    {"KuDNN::Element::TypeT::BF16", KuDNN::Element::TypeT::BF16},
    {"KuDNN::Element::TypeT::S32", KuDNN::Element::TypeT::S32},
    {"KuDNN::Element::TypeT::S8", KuDNN::Element::TypeT::S8},
    {"KuDNN::Element::TypeT::U8", KuDNN::Element::TypeT::U8},
};

bool kudnn_conv_01();
bool kudnn_conv_02();
bool kudnn_gnorm_01();
bool kudnn_lnorm_01();
bool kudnn_linearActivation_01();
bool kudnn_linearRes_01();
bool kudnn_rnorm_01();
bool kudnn_softmax_01();