#!/bin/bash

set -e

rm -rf build
mkdir build
cd build

mpicc ../mpi_send_recv.c -o mpi_send_recv
mpicc ../mpi_reduce.c -o mpi_reduce
mpicc ../mpi_gather.c -o mpi_gather
mpicc ../mpi_scatter.c -o mpi_scatter
mpicc ../mpi_alltoall.c -o mpi_alltoall