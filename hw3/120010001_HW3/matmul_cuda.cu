#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"


#define THREADS_PER_BLOCK 32
#ifndef N
#define N 10
#endif

__global__ void matmul_two(float *A, float *B, float *C);
__global__ void matmul_one(float *A, float *B, float *C, int row);
void matmul_caller_two(float *A_dev, float *B_dev, float*C_dev, float *A, float *B, float *C, int size);
void matmul_caller_one(float *A_dev, float *B_dev, float *C_dev, float *A, float *B, float *C, int size);

int main()
{
	// host copies of A, B & C
	float *A, *B, *C;
	// device copies of A, B & C
	float *A_dev, *B_dev, *C_dev;


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
	for (int i = 0; i < N*N; ++i)
	{
		A[i] = (float)1;
		B[i] = (float)1;
		C[i] = (float)0;
	}
	int c = 2;
	if (c == 2)
	{
		matmul_caller_two(A_dev, B_dev, C_dev, A, B, C, size);
	}
	else if (c==1)
	{
		matmul_caller_one(A_dev, B_dev, C_dev, A, B, C, size);
				
	}
	else
	{
		printf("enter correct case code [1,2]!\n");
		free(A);
		free(B);
		free(C);

		cudaFree(A_dev);
		cudaFree(B_dev);
		cudaFree(C_dev);

		return 1;
	} 

 	//for (int i=0; i<N*N; i++)
	//{
	//	printf("C[%d]=%f\n",i,C[i]);
	//}
	printf("C[last]=%f\n", C[0]);
	free(A);
	free(B);
	free(C);

	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(C_dev);

	return 0;

}



__global__ void matmul_one(float *A, float *B, float *C, int row)
{	
	
	__shared__ float temp[THREADS_PER_BLOCK];
	//printf("N=%d\n", N);
	//printf("row = %d\n", row);
	for (int col = 0; col<N; col++) // iteration for single row of C 
	{	
		//printf("--------------col = %d----Blockid = %d-------thread = %d----\n", col, blockIdx.x, threadIdx.x);
		int index = blockIdx.x*blockDim.x + threadIdx.x; // threads compute the product of elements in row of A and column of B
		temp[threadIdx.x] = A[index + row*N]*B[col + index*N];
		//printf("temp[%d] = A[%d + %d*%d]*B[%d + %d*%d]=%f*%f=%f\n", index, index,row, N,  col, index, N, temp[index], A[index + row*N], B[col + index*N]);
		
		__syncthreads();
		
			
		if (0 == threadIdx.x)
		{
			float sum = 0;
			for (int j = 0; j < THREADS_PER_BLOCK; j++)
			{
				sum += temp[j];
		//		printf("???? temp[%d] = %f ???? blockid = %d ??????\n",j,  temp[j], blockIdx.x);
					
			}
		//	printf("???? ddot = %f ???? blockid = %d ??????\n", sum, blockIdx.x);
			atomicAdd(&C[row*N + col], sum);
			
		}
	}

}



__global__ void matmul_two(float *A, float *B, float *C)
{
	float ddot = 0;
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	if (row < N && col < N)
	{
		for (int i = 0; i < N; ++i)
		{
			ddot += A[row*N + i] * B[col + i*N];
		}
	}
	C[row * N + col] = ddot;
}


void  matmul_caller_two(float *A_dev, float *B_dev, float *C_dev, float *A, float *B, float *C, int size)
{
	cudaMemcpy(A_dev, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_dev, B, size, cudaMemcpyHostToDevice);
	
	// First allocate 1 block in each direction of the matrix i.e. 1 block in total
	dim3 blocksPerGrid(1,1);

	dim3 threadsPerBlock(N, N);
	
 	
	// Then check if the total number of elements in the matrix is greater than 512
	// which is the maximum allowable limit of THREADS_PER_BLOCK
	if (N*N > THREADS_PER_BLOCK*THREADS_PER_BLOCK)
	{	
		threadsPerBlock.x = THREADS_PER_BLOCK;
		threadsPerBlock.y = THREADS_PER_BLOCK;
		blocksPerGrid.x = ceil((float)N/(float)threadsPerBlock.x);
		blocksPerGrid.y = ceil((float)N/(float)threadsPerBlock.y);
	
	}
	
	matmul_two<<<blocksPerGrid, threadsPerBlock>>>(A_dev, B_dev, C_dev);
	cudaMemcpy(C, C_dev, size, cudaMemcpyDeviceToHost);
	
}

void matmul_caller_one(float *A_dev, float *B_dev, float *C_dev, float *A, float *B, float *C, int size)
{	
	cudaMemcpy(B_dev, B, size, cudaMemcpyHostToDevice);
	cudaMemcpy(A_dev, A, size, cudaMemcpyHostToDevice);
	
	int num_blocks = 1;
	int num_threads = N;
	if (N > THREADS_PER_BLOCK)
	{	
		num_threads = THREADS_PER_BLOCK;
		num_blocks = ceil((float)N/(float)THREADS_PER_BLOCK);
	}
	
	for(int i=0; i<N; i++)
	{	
		matmul_one<<<num_blocks, num_threads>>>(A_dev, B_dev, C_dev, i);
	}
	
	cudaMemcpy(C, C_dev, size, cudaMemcpyDeviceToHost);

}
