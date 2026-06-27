#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char** argv) {
    int rank, size;
    int result = 0;
    int *send_data;

    // 初始化MPI环境
    MPI_Init(&argc, &argv);
    // 获取当前进程编号
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // 获取总进程数
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // 使用当前时间作为随机数种子（确保每次运行结果不同）
    srand((unsigned int)time(NULL));

    // 生成一个随机整数（范围：0 ~ RAND_MAX，通常是32767）
    int random_num[1024];

    for(int i = 0; i < 1024; i++)
    {
        random_num[i] = rand();
    }

    // 检查是否小于1024个进程（本示例仅支持1024进程通信）
    if (size > 1024) {
        if (rank == 0) {
            printf("ERROR\n");
        }
        MPI_Finalize();
        return 1;
    }

    int recv_data = 0;

    // 根进程（进程0）准备要分发的数据
    if (rank == 0) {
        send_data = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            send_data[i] = random_num[i];
        }
    }

    // 根进程向所有进程分发数据（包括自身）
    MPI_Scatter(
        send_data,       // 发送缓冲区（仅根进程有效）
        1,               // 每个进程接收的数据数量
        MPI_INT,         // 数据类型
        &recv_data,      // 接收缓冲区（每个进程的本地变量）
        1,               // 每个进程接收的数据数量
        MPI_INT,         // 接收数据类型
        0,               // 根进程编号（发送数据的进程）
        MPI_COMM_WORLD   // 通信域
    );

    if(recv_data != random_num[rank]){
        int local_result = 1;
        // 汇总所有进程的local_value到进程0，进行求和操作
        MPI_Reduce(
                &local_result,  // 发送缓冲区（本地数据）
                &result,        // 接收缓冲区（仅根进程有效）
                1,              // 数据数量
                MPI_INT,        // 数据类型
                MPI_SUM,        // 汇总操作（求和）
                0,              // 根进程编号（接收结果的进程）
                MPI_COMM_WORLD  // 通信域
        );
    }


    // 只有根进程（进程0）有汇总结果
    if (rank == 0) {
        if(result == 0)
        {
                printf("PASSED\n");
        } else {
                printf("ERROR\n");
        }
    }

    if(rank == 0)
    {
        free(send_data);
    }

    // 结束MPI环境
    MPI_Finalize();
    return 0;
}
