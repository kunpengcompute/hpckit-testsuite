cputype=$(cat /proc/cpuinfo | grep 'CPU part' | awk 'NR==1{print $4}')
if [[ "$cputype" == "0xD01" ]]; then
    echo "kulitho: Not suppot on this platform" 
else
    cd ./test
    ./test_kulitho
fi
