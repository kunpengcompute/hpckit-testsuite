#!/bin/bash
#
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
#
set -e

cputype=$(cat /proc/cpuinfo | grep 'CPU part' | awk 'NR==1{print $4}')

if [[ "$cputype" == "0xd22" ]]; then
  ./benchmark 128 128 128
fi

