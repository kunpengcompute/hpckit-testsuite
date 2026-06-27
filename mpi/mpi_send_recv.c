#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    int rank, size;

    // 使用当前时间作为随机数种子（确保每次运行结果不同）
    srand((unsigned int)time(NULL));

    // 生成一个随机整数（范围：0 ~ RAND_MAX，通常是32767）
    int random_num = rand();

    // 初始化MPI环境
    MPI_Init(&argc, &argv);
    // 获取当前进程编号
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // 获取总进程数
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 检查是否只有2个进程（本示例仅支持2进程通信）
    if (size != 2) {
        if (rank == 0) {
            printf("ERROR\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        // 进程0发送数据
        int data = random_num;  // 要发送的数据
        MPI_Send(&data,       // 数据缓冲区地址
                 1,           // 数据数量
                 MPI_INT,     // 数据类型
                 1,           // 目标进程编号
                 0,           // 消息标签
                 MPI_COMM_WORLD);  // 通信域
    } else if (rank == 1) {
        // 进程1接收数据
        int data;
        MPI_Recv(&data,       // 接收数据的缓冲区地址
                 1,           // 接收数据数量
                 MPI_INT,     // 数据类型
                 0,           // 源进程编号
                 0,           // 消息标签（需与发送方一致）
                 MPI_COMM_WORLD,  // 通信域
                 MPI_STATUS_IGNORE);  // 忽略状态信息
        if (data == random_num) {
                printf("PASSED\n");
        } else {
                printf("ERROR\n");
        }
    }

    // 结束MPI环境
    MPI_Finalize();
    return 0;
}
