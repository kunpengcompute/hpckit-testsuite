# kupl-test

#### 介绍
kupl_test 提供了一组使用 kupl 库的用例

#### 软件架构
test目录下包含5个用例

test_memory: 数据拷贝

test_mma: 矩阵编程

test_sgraph 静态图编程

test_parallel_for 并行 for 循环
    
test_queue 多队列编程

#### 编译方法

执行以下命令

sh build.sh --compiler=clang

#### 运行方法

执行以下命令

sh run.sh

#### 运行结果

运行输出结果：

所有用例运行结果均为PASSED
[  PASSED  ] x tests.