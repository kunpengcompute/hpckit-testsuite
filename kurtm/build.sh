set -e

cputype=$(cat /proc/cpuinfo | grep 'CPU part' | awk 'NR==1{print $4}')
if [[ "$cputype" == "0xd22" ]]; then
  echo "build KuRTM testsuite.."
  rm -rf benchmark
  rm -rf Makefile
  rm -rf CMake*
  rm -rf cmake_install.cmake
  cmake ./test 
  make 
  echo "build KuRTM testsuite done"
else
  echo "Not suppot on this platform"
fi


