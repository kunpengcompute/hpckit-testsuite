cputype=$(cat /proc/cpuinfo | grep 'CPU part' | awk 'NR==1{print $4}')
    if [[ "$cputype" == "0xd22" ]]; then
        cd test
        mpicxx -lkuqcd test_solver.cpp -o ../test_solver
        cd ..
    fi
