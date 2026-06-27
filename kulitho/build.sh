#!/bin/bash

cputype=$(cat /proc/cpuinfo | grep 'CPU part' | awk 'NR==1{print $4}')
if [[ "$cputype" == "0xD01" ]]; then
  echo "kulitho: Not suppot on this platform" 
else
  if [ -z "${HPCKIT_PATH}" ]; then
      echo "Error: Environment variable 'HPCKIT_PATH' is not set!"
      echo "Please configure HPCKIT_PATH to point to the HPCKit installation directory, for example:"
      echo "    export HPCKIT_PATH=/opt/HPCKit"
      echo "Then re-run the build command."
      exit 1
  fi
  echo "HPCKIT_PATH=${HPCKIT_PATH}"
  case "$1" in
      gcc)
          export CC=$(which gcc)
          export CXX=$(which g++)
          ;;
      bisheng)
          export CC=$(which clang)
          export CXX=$(which clang++)
          ;;
      *)
          echo "Unsupported compiler $1."
          exit 1
          ;;
  esac
  cd ./test
  make clean
  make 
fi
