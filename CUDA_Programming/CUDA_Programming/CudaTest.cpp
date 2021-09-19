#include <stdio.h>
#include "CudaTest.cuh"

int sum_int(int a, int b) {
	return a + b;
}

int multiply_int(int a, int b) {
	return a * b;
}

int main() {
	int a = 3, b = 4, c = 0, cu = 0;
	int cpu_m = 0, gpu_m = 0;
	
	c = sum_int(a, b);
	cpu_m = multiply_int(a, b);

	CudaTest gpuacc;
	gpuacc.sum_cuda(a, b, &cu);
	gpuacc.multiply_cuda(a, b, &gpu_m);

	printf("CPU : %d, %d\n", c, cpu_m);
	printf("GPU : %d, %d\n", cu, gpu_m);

	return 0;
}


