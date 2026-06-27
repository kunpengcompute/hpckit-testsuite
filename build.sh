#!/bin/bash

for dir in */; do
    dir_name="${dir%/}"

    if [ -f "$dir_name/build.sh" ]; then
        echo "=== 构建 $dir_name ==="

        cd "$dir_name"

        sh ./build.sh $1

        cd ..

        echo "=== 构建 $dir_name 完成 ==="
    else 
        echo "=== 跳过 $dir_name:未找到 build.sh ==="
        echo
    fi
done

echo "所有组件测试构建完毕！"