#include <stdio.h>
#include "CudaTest.cuh"

int sum_int(int a, int b) {
	return a + b;
}

int multiply_int(int a, int b) {
	return a * b;
}

void print_loop() {
	for (int b_x = 0; b_x < 2; b_x++) {
		for (int b_y = 0; b_y < 3; b_y++) {
			for (int b_z = 0; b_z < 4; b_z++) {
				for (int t_x = 0; t_x < 3; t_x++) {
					for (int t_y = 0; t_y < 3; t_y++) {
						for (int t_z = 0; t_z < 3; t_z++) {

							int block_id =
								b_x +
								b_y * 2 +
								b_z * 2 * 3;

							int block_offset =
								block_id * 24;

							int thread_offset =
								t_x +
								t_y * 3 +
								t_z * 9;

							int id = block_offset + thread_offset;
							printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n", id, b_x, b_y, b_z, block_offset, t_x, t_y, t_z, thread_offset);
						}
					}
				}
			}
		}
	}

}

int main() {
	int a = 3, b = 4, c = 0, cu = 0;
	int cpu_m = 0, gpu_m = 0;
	
	c = sum_int(a, b);
	cpu_m = multiply_int(a, b);
	print_loop();

	CudaTest gpuacc;
	//gpuacc.sum_cuda(a, b, &cu);
	//gpuacc.multiply_cuda(a, b, &gpu_m);
	gpuacc.thread_debug();

	printf("CPU : %d, %d\n", c, cpu_m);
	printf("GPU : %d, %d\n", cu, gpu_m);

	return 0;
}


