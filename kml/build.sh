#!/bin/bash
# export HPCKIT_PATH=/opt/HPCKit/
if [ -z "${HPCKIT_PATH}" ]; then
    echo "Error: Environment variable 'HPCKIT_PATH' is not set!"
    echo "Please configure HPCKIT_PATH to point to the HPCKit installation directory, for example:"
    echo "    export HPCKIT_PATH=/opt/HPCKit"
    echo "Then re-run the build command."
    exit 1
fi

echo "HPCKIT_PATH=${HPCKIT_PATH}"
cd ./test
make clean
cputype=$(cat /proc/cpuinfo | grep 'CPU part' | awk 'NR==1{print $4}')
if [[ "$cputype" == "0xd01" ]]; then
    make IS_NEED_CONV=0
else
    make IS_NEED_CONV=1
fi