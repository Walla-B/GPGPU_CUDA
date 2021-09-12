#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

extern void addWithCuda(int *c, const int *a, const int *b, unsigned int size);

int main() {


