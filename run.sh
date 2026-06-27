#!/bin/bash

for dir in */; do
    dir_name="${dir%/}"

    if [ -f "$dir_name/run.sh" ]; then
        echo "=== 执行 $dir_name 测试 ==="

        cd "$dir_name"

        sh ./run.sh

        cd ..

        echo "=== 测试 $dir_name 完成 ==="
    else 
        echo "=== 跳过 $dir_name:未找到 run.sh ==="
        echo
    fi
done

echo "所有组件测试执行完毕！"