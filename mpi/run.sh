#!/bin/bash

# 初始化总统计变量
total_success=0
total_error=0

# 定义测试列表
tests=("-np 2 ./build/mpi_send_recv" "-np 8 ./build/mpi_reduce" "-np 8 ./build/mpi_gather" "-np 8 ./build/mpi_scatter" "-np 8 ./build/mpi_alltoall")

echo "=== 执行 mpi 测试 ==="

# 循环运行每个测试
for test in "${tests[@]}"; do
    # 步骤1：按空格分割字符串为数组（获取所有参数）
    IFS=' ' read -ra parts <<< "$test"

    # 步骤2：获取数组最后一个元素（即程序的完整路径）
    full_path="${parts[-1]}"

    # 步骤3：从完整路径中提取文件名（去掉目录部分）
    test_name=$(basename "$full_path")

    echo "start test $test_name"

    # 运行测试并捕获输出（保存到临时变量）
    output=$(mpirun --allow-run-as-root $test 2>&1)  # 2>&1 确保错误输出也被捕获

    # 匹配 "success" 相关的数字（不区分大小写）
    success=$(echo "$output" | grep -oi 'PASSED' | wc -l)
    # 匹配 "error" 相关的数字（不区分大小写）
    error=$(echo "$output" | grep -oi 'ERROR' | wc -l)

    # 累加统计
    if [ $error -eq 0 ] && [ $success -eq 1 ]; then
        echo "end test $test_name PASSED"
        total_success=$((total_success + success))
    else
        echo "end test $test_name FAILED"
        total_error=$((total_error + error))
    fi
done

# 输出总统计结果
echo "$((total_success + total_error)) tests ran, $total_success tests PASSED, $total_error tests FAILED."
echo "============================"
