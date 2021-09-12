#include <stdio.h>
#include "CudaTest.cuh"

int sum_int(int a, int b) {
	return a + b;
}

int main() {
	int a = 3, b = 4, c = 0, cu = 0;

	c = sum_int(a, b);

	CudaTest gpuacc;
	gpuacc.sum_cuda(a, b, &cu);

	printf("CPU : %d\n", c);
	printf("GPU : %d\n", cu);

	return 0;
}