#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"


#define THREADS_PER_BLOCK 512

__global__ void matmul(float* A, float* B, float*C);

int main()
{
	int N = 5;
	
	// host copies of A, B & C
	float *A, *B, *C, N, C_elem;
	// device copies of A, B & C
	float *A_dev, *B_dev, *C_dev, N_dev;

	int size = N*N*sizeof(float);

	// allocating memories on the device for the matrices
	cudaMalloc((void**)&A_dev, size);
	cudaMalloc((void**)&B_dev, size);
	cudaMalloc((void**)&C_dev, size);

	// allocating memory on the host for the matrices
	A = (float*)malloc(size);
	B = (float*)malloc(size);
	C = (float*)malloc(size);

	// initialise A and B
	int i;
	for (int i = 0; i < N*N; ++i)
	{
		A[i] = (float)1;
		B[i] = (float)1;
		C[i] = (float)0;
	}

	cudaMemcpy(&N_dev, &N, 1, cudaMemcpyHostToDevice);
	cudaMemcpy(A_dev, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_dev, B, size, cudaMemcpyHostToDevice);
	// cudaMemcpy(C_dev, C, size, cudaMemcpyHostToDevice);

	// First allocate 1 block in each direction of the matrix i.e. 1 block in total
	dim3 blocksPerGrid(1,1);

	dim3 threadsPerBlock(N, N);
	
	// Then check if the total number of elements in the matrix is greater than 512
	// which is the maximum allowable limit of THREADS_PER_BLOCK
	if (N*N > THREADS_PER_BLOCK)
	{	
		threadsPerBlock.x = THREADS_PER_BLOCK;
		threadsPerBlock.y = THREADS_PER_BLOCK;
		blocksPerGrid.x = ceil((float)N/(float)threadsPerBlock.x);
		blocksPerGrid.y = ceil((float)N/(float)threadsPerBlock.y);
	}

	matmul<<<blocksPerGrid, threadsPerBlock>>>matmul(A_dev, B_dev, C_dev, N);

	cudaMemcpy(&C_elem, &C_dev[N*N-1], 1, cudaMemcpyDeviceToHost);

	free(A);
	free(B);
	free(C);

	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(C_dev);

	return 0;

}

__global__ void matmul(float *A, float *B, float *C, float N)
{
	float ddot = 0;
	int i;

	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	for (int i = 0; i < N; ++i)
	{
		ddot += A[row*N + i] * B[col + i*N];
	}

	C[row * N + col] = ddot;
}