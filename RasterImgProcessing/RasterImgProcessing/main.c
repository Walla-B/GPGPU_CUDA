#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

extern void addKernel(int* c, const int* a, const int* b);

int main() {
	int arr_c[3] = { 0, 0 ,0 };
	const int arr_b[3] = { 10, 20, 30 };
	const int arr_a[3] = { 1,2,3 };

	addKernel(arr_c, arr_a, arr_b);
	return 0;

}