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