#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    int rank, size;
    int global_sum = 0;
    // 使用当前时间作为随机数种子（确保每次运行结果不同）
    srand((unsigned int)time(NULL));

    // 生成一个随机整数（范围：0 ~ RAND_MAX，通常是32767）
    int random_num[1024];

    for(int i = 0; i < 1024; i++)
    {
        random_num[i] = rand();
    }

    // 初始化MPI环境
    MPI_Init(&argc, &argv);
    // 获取当前进程编号
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // 获取总进程数
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 检查是否小于1024个进程（本示例仅支持1024进程通信）
    if (size > 1024) {
        if (rank == 0) {
            printf("ERROR\n");
        }
        MPI_Finalize();
        return 1;
    }

    int local_value = random_num[rank];

    // 汇总所有进程的local_value到进程0，进行求和操作
    MPI_Reduce(
        &local_value,   // 发送缓冲区（本地数据）
        &global_sum,    // 接收缓冲区（仅根进程有效）
        1,              // 数据数量
        MPI_INT,        // 数据类型
        MPI_SUM,        // 汇总操作（求和）
        0,              // 根进程编号（接收结果的进程）
        MPI_COMM_WORLD  // 通信域
    );

    // 只有根进程（进程0）有汇总结果
    if (rank == 0) {
        int result = 0;
        for(int i = 0; i < size; i++)
        {
                result += random_num[i];
        }
        if(result == global_sum)
        {
                printf("PASSED\n");
        } else {
                printf("ERROR\n");
        }
    }

    // 结束MPI环境
    MPI_Finalize();
    return 0;
}
