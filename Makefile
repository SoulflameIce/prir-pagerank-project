# Makefile for PageRank project: OpenMP, MPI, CUDA
CC = gcc
MPICC = mpicc
NVCC = nvcc

CFLAGS = -O3 -march=native
OMPFLAGS = -fopenmp
LDFLAGS =

all: openmp mpi cuda

openmp: pagerank_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o pagerank_openmp pagerank_openmp.c $(LDFLAGS)

mpi: pagerank_mpi.c
	$(MPICC) -O3 -o pagerank_mpi pagerank_mpi.c

cuda: pagerank_cuda.cu
	$(NVCC) -O3 -arch=sm_60 -o pagerank_cuda pagerank_cuda.cu

clean:
	rm -f pagerank_openmp pagerank_mpi pagerank_cuda *.rvec *.stats

.PHONY: all openmp mpi cuda clean
