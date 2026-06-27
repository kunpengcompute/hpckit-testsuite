cputype=$(cat /proc/cpuinfo | grep 'CPU part' | awk 'NR==1{print $4}')
if [[ "$cputype" == "0xd22" ]]; then
    if [ -e ./bin/test_kutacc ]; then
        ./bin/test_kutacc
    fi
fi
