#include "CudaTest.cuh"
#include <stdio.h>

CudaTest::CudaTest(void) {

}

CudaTest::~CudaTest() {

}

__global__ void sum_kernel(int a, int b, int* c) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	c[tid] = a + b;
}
__global__ void multiply_kernel(int a, int b, int* c) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	c[tid] = a * b;
}

__global__ void thread_indexing_kernel() {
	int block_id =
		blockIdx.x +
		blockIdx.y * gridDim.x +
		blockIdx.z * gridDim.x * gridDim.y;

	int block_offset =
		block_id *
		blockDim.x * blockDim.y * blockDim.z;

	int thread_offset =
		threadIdx.x +
		threadIdx.y * blockDim.x +
		threadIdx.z * blockDim.x * blockDim.y;

	int id = block_offset + thread_offset;

	printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n", id, blockIdx.x, blockIdx.y, blockIdx.z, block_offset, threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);

}

int CudaTest::sum_cuda(int a, int b, int* c) {
	int* f;
	cudaMalloc((void**)&f, sizeof(int) * 1);
	cudaMemcpy(f, c, sizeof(int) * 1, cudaMemcpyDeviceToDevice);

	sum_kernel << <1, 1 >> > (a, b, f);
	cudaMemcpy(c, f, sizeof(int) * 1, cudaMemcpyDeviceToHost);

	cudaFree(f);

	return true;
}

int CudaTest::multiply_cuda(int a, int b, int* c) {
	int* g;
	cudaMalloc((void**)&g, sizeof(int) * 1);
	cudaMemcpy(g, c, sizeof(int) * 1, cudaMemcpyDeviceToDevice);

	multiply_kernel << <1, 1 >> > (a, b, g);
	cudaMemcpy(c, g, sizeof(int) * 1, cudaMemcpyDeviceToHost);

	cudaFree(g);

	return true;
}

int CudaTest::thread_debug() {
	const int b_x = 2, b_y = 3, b_z = 4;
	const int t_x = 3, t_y = 3, t_z = 3;

	int blocks_per_grid = b_x * b_y * b_z;
	int threads_per_block = t_x * t_y * t_z;

	dim3 blocksPerGrid(b_x, b_y, b_z);
	dim3 threadsPerBlock(t_x, t_y, t_z);

	thread_indexing_kernel <<<blocksPerGrid, threadsPerBlock>>>();
	return true;
}

