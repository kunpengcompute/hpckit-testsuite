#!/bin/bash
export DNN_INC=${HPCKIT_PATH}/latest/kudnn/$1/include
export DNN_LIB=${HPCKIT_PATH}/latest/kudnn/$1/lib
export BLAS_LIB=${HPCKIT_PATH}/latest/kudnn/$1/lib/matrix/kblas/multi
cputype=$(cat /proc/cpuinfo | grep 'CPU part' | awk 'NR==1{print $4}')
if [[ "$cputype" == "0xd22" ]]; then
    echo "CXX=${CXX}"
    echo "DNN_INC=${DNN_INC}"
    echo "DNN_LIB=${DNN_LIB}"
    echo "BLAS_LIB=${BLAS_LIB}"
    ${CXX} src/function.cpp src/main.cpp -o function_test \
        -I ${DNN_INC} \
        -L ${DNN_LIB} \
        -L ${BLAS_LIB} \
        -lkblas \
        -lkudnn

else
  echo "kudnn: Not suppot on this platform"
fi
