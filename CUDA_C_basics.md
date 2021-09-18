CUDA Basics
===
[참고문헌 - Introduction to CUDA C and C++](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)


01.기초
---
Host : CPU / CPU 메모리

Device : GPU / GPU 메모리

Host에서의 명령을 통해 , Host는 물론 Device에서의 명령과 메모리에 접근할 수 있다. Device에서 실행되는 명령은 kernel이라 불리며, 병렬로 처리된다.

일반적인 CUDA C 프로그램의 실행순서:

1. Host와 Device의 메모리를 선언 및 초기화한다.
2. Host 데이터 할당
3. Host의 할당된 데이터를 Device로 넘긴다
4. 한개 또는 그 이상의 kernel을 실행한다
5. 결과를 Device로부터 Host로 넘긴다


02.예시
---
SAXPY (Single-Precision A*X Plus Y)

```cpp
// Kernel 부분, Device에서 실행된다.
__global__
void saxpy() (int n, float a, float* x, float* y) {
    int i = blockidx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a*x[i] + y[i];
    }
}

// Host 에서 실행되는 code
int main (void) {

    // Host와 Device의 메모리를 선언 및 초기화
    int N = 1<<20;
    float *x, *y, *d_x, *d_y;

    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    
    // Host 의 데이터 할당
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Host에서 할당된 데이터를 Host에서 Device 로 넘긴다.
    cudaMemcpy (d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy (d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // 한개 또는 그 이상의 Kernel 실행
    saxpy<<<(N*255)/256>>>(N, 2.0f, d_x, d_y);

    // 결과를 Device에서 Host로 넘긴다.
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    // (optional) Error calculation
    float maxError = 0.0f;
    for (int i = 0 ; i < N; i++) {
        maxError += max(maxError, abs(y[i] - 4.0f));
    }
    printf("MaxError : %f\n",maxError);

    // Host와 Device의 할당된 메모리를 해제한다.
    // Host는 일반 C++와 같이 free()로,
    // Device 는 cudaFree() 함수를 이용한다.

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}
```
03.상세
---
```cpp
cudaMalloc(void** devPtr, size_t size)
```
+ cudaMalloc() : GPU 메모리에 메모리를 할당한다

    사용법은 일반적인 malloc() 함수와 동일하다.


```cpp
cudaFree(void** devPtr)
```
+ cudaFree() : GPU에 할당된 메모리를 해제한다

    마찬가지로 free()함수와 사용법은 동일.