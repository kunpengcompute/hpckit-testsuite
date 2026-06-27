#include <functional>
#include "function.hpp"

std::unordered_map<std::string, std::function<bool()>> funcMap;
int main(int argc, char **argv)
{
    funcMap["kudnn_conv_01"] = kudnn_conv_01;
    funcMap["kudnn_conv_02"] = kudnn_conv_02;
    funcMap["kudnn_gnorm_01"] = kudnn_gnorm_01;
    funcMap["kudnn_lnorm_01"] = kudnn_lnorm_01;
    funcMap["kudnn_linearActivation_01"] = kudnn_linearActivation_01;
    funcMap["kudnn_linearRes_01"] = kudnn_linearRes_01;
    funcMap["kudnn_rnorm_01"] = kudnn_rnorm_01;
    funcMap["kudnn_softmax_01"] = kudnn_softmax_01;

    int testNum = 0;
    int passNum = 0;
    
    if (argc == 2) {
        testNum ++;
        if (funcMap.count(argv[1]) > 0) {
            std::cout << "start test " << argv[1] << std::endl;
            bool ret = funcMap[argv[1]]();
            if (!ret) {
                std::cout << "end test " << argv[1] << " FAILED" << std::endl;
            } else {
                std::cout << "end test " << argv[1] << " PASSED" << std::endl;
                passNum ++;
            }
        } else {
            std::cout << "No test named " << argv[1] << std::endl;
        }
    } else if (argc == 1) {
        for (auto &func : funcMap) {
            testNum++;
            std::cout << "start test " << func.first << std::endl;
            bool ret = func.second();   
            if (!ret) {
                std::cout << "end test " << func.first << " FAILED" << std::endl;
            } else {
                std::cout << "end test " << func.first << " PASSED" << std::endl;
                passNum ++;
            }
        }
    }
    
    std::cout << testNum << " tests ran, " << passNum << " tests PASSED, " << (testNum - passNum) << " tests FAILED." << std::endl;

    return 0;
}