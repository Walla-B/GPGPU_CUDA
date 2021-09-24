CUDA Basics
===
[참고문헌 - Introduction to CUDA C and C++](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)


01.기초
---
Host : CPU / CPU 메모리

Device : GPU / GPU 메모리

Host에서의 명령을 통해 , Host는 물론 Device에서의 명령과 메모리에 접근할 수 있다. Device에서 실행되는 명령은 kernel이라 불리며, 병렬로 처리된다.


+ 일반적인 CUDA C 프로그램의 실행순서:

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
    saxpy<<<(N*255)/256, 256>>>(N, 2.0f, d_x, d_y);

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
cudaMalloc(void** devPtr, size_t size);
```
+ cudaMalloc() : GPU 메모리에 메모리를 할당한다

    사용법은 일반적인 malloc() 함수와 동일하다.


```cpp
cudaFree(void** devPtr);
```
+ cudaFree() : GPU에 할당된 메모리를 해제한다

    마찬가지로 free()함수와 사용법은 동일.


> Q. 왜 더블포인터 (void**)를 사용하는가?
>
> A. 일반적인 malloc() 함수와의 차이점으로, malloc()은 할당된 메모리 공간의 주소를 반환하는 반면에,
>
> cudaMalloc() 은 cudaError_t 값을 반환한다. (성공시 CudaSuccess, 실패시 Cudafail)
> 
> 한편, C에는 Call by reference가 존재하지 않으므로 cudaError_t 값에대한 참조를 반환하기 위해
>
> 이중 포인터가 사용되는 것이다.
> 
> [참고](https://stackoverflow.com/questions/7989039/use-of-cudamalloc-why-the-double-pointer)


```cpp
cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
``` 
+ cudaMemcpy() : Device와 Host간 데이터를 복사 전달한다

    Memcpy()와 비슷하지만 네번째 인자로 방향을 결정한다. 
    
    (cudaMemcpyHostToDevice : Host to Device, cudaMemcpyDeviceToHost : Device to Host)

```cpp
// Kernel declaration
__global__ void Func(float* param);

// Kernel execution
Func<<< Dg, Db, Ns >>>(param);
```

> + CUDA Keywords:
> 
>   ```cpp
>   // GPU에서 동작하는 함수, CPU에서 호출 가능
>   __global__ void Func(float* param);
> 
>   // 오직 CPU에서만 동작 가능한 함수
>   __host__ void Func(float* param);
> 
>   // GPU 내부에서 동작하며, GPU에서만 호출 가능한 함수
>   __device__ void Func(float* param);
> 
>   // CPU, GPU 각각 모두 호출 가능한 함수
>   __host device__ void Func(float* param);
>   ```
>  
  
+   Dg : (dim3) 실행되는 블록의 개수 (= Grid의 크기)

    Db : (dim3) 블럭당 실행되는 스레드의 개수 (= 블럭의 크기)

    Ns : (size_t) optional, 기본값은 0이며 블럭당 할당되는 동적 메모리의 양이다

    [참고](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)

> + dim3 : 차원을 결정하는 벡터타입 변수. 초기화하지 않은 부분은 1로 초기화된다.
>
>   ```cpp
>   // 3차원
>   dim3 dimention(uint x, uint y, uint z);
>
>   // 2차원
>   dim3 diention(uint x, uint y);
> 
>   // 1차원
>   dim3 diention(uint x);
>   ```



> + Block And Threads :
> 
>   ![img](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/grid-of-thread-blocks.png)
>
>   한개의 Grid는 여러개의 Block으로, 또 각각의 Block 들은 여러개의 Threads를 가지고 있어 병렬처리를 가능하게 한다.
>
>   위의 이미지에서는 Block과 Thread 모두 이차원으로 정의되었으며, 접근시에도 이차원 인덱스를 이용할 것이다.
>
>   ```cpp
>   // 위의 삽도에서 정의된 대로 Block 및 Thread의 dim3 구현방법
>   dim3 threadsPerBlock(3,4);
>   dim3 numBlocks(2,3);
>   ```
>   그러나 위와 같은 형태는 주어진 데이터 크기에 따라 유동적으로 바뀔 수 없으므로, 앞서 초기화한
> 
>   메모리의 크기를 이용해 다음과 같이 구현한다.
> 
> + 일반적인 형태의 dim3 구현 방법
>   ```cpp
>   // 최적 Block별 Thread 개수
>   dim3 threadsPerBlock(16, 16);
> 
>   // 연산이 필요한 데이터가 float data[N][M]과 같이 주어졌을 경우,
>   // 데이터를 Block의 크기 (=threadsPerBlock) 의 단위로 쪼갤수도 있다.
>   dim3 numBlocks( (N / threadsPerBlock.x) , (M / threadsPerBlock.y) );
>
>   // Kernel 실행
>   Func<<<numBlocks, threadsPerBlock>>>(param);
>   ```


```cpp
__global__
void saxpy() (int n, float a, float* x, float* y) {
    int i = blockidx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a*x[i] + y[i];
    }
}
``` 
+ Kernel 내에서의 인덱싱 키워드
  
    |키워드|타입|설명|비고|
    |:---:|:---:|:--:|:--:|
    |gridDim|dim3|Grid가 포함한 Block들의 치수| = Dg
    |blockIdx|uint3|Grid 안에서의 Block별 고유 Index|
    |blockDim|dim3|Block이 포함한 Thread들의 치수| = Db
    |threadIdx|uint3|Block 안에서의 Thread별 고유 Index|

>gridDim의 각 원소 x, y, z 를 모두 곱하면 총 block 들의 개수가,
>
>blockDim의 각 원소 x, y, z 를 모두 곱하면 한 블록마다 가지고있는 Thread들의 개수가 구해진다.

위의 키워드들 중, 병렬처리를 위한 메모리 접근에 필요한 index를 계산하기 위해 일반적으로

blockIdx, blockDim, threadIdx가 필요하며, 이를 이용해 index를 구한다.


> indexing 방법:
> + Block, Thread가 최대 N차원일때, Cartesian Method 와 비슷하게 N개의 index 나타내는 방법
>   ```cpp
>   int index1 = blockDim.x * blockIdx.x + threadIdx.x;
>   int index2 = blockDim.y * blockIdx.y + threadIdx.y; //최대 2차원인 경우 추가
>   int index3 = blockDim.z * blockIdx.z + threadIdx.z; //최대 3차원인 경우 추가
>   ```
> + Block이 1차원, Thread가 2차원일때 한개의 index로 나타내는 방법
>   ```cpp
>   int index = blockIdx.x * (blockDim.x * blockDim.y)
>                   + threadIdx.y * blockDim.x + threadIdx.x;
>   ```
> + 이외 N차원 Block, M차원 Thread를 하나의 index로 나타내는 방법은 다음을 참고:
> 
>   [차원별 Index 맵핑 방법](https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf)

+ Kernel 작성 시 주의해야 할 부분들 
 
    [참고 - Control Flow best practicies](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#control-flow)

    GPU는 기본적으로 SIMT (Single Instruction Multiple Threads) 을 따른다.

    이것이 의미하는 것은, 한개의 스레드가 어떠한 분기문을 타고 다른 Control Path로 들어가게 된다면

    Block으로 묶여있는 다른 Thread들도 좋든 싫든 같은 분기문을 타고 들어가야 한다는 것이다.

    그렇게 조건에 맞지 않는 분기문을 타고 들어간 다른 Thread들은 결과값을 무시하고 다시 연산하는 방법으로

    값을 구하기 때문에 분기문이 들어간 Kernel들은 그렇지 않은 경우에 비해 throughput이 낮아질 수 있다.


    ```cpp
    // Bad implemetations

    __global__
    void FOO(int n,  float* a, float* b) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        // if, switch, do, for, while 등등 분기문들

        if ( /* Check_Conditions */ ) {
            // Do something
        }
        else if ( /* Check_ Conditions */ ) {
            // Do something
        }
        else {
            // Do something
        }

        b[i] = n*a[i];
    }

    __host__
    ```

> Q. 그렇다면 해결법은?
>
> A.
>> 1. 분기가 많이 들어가는 경우에 한해서 분기문을 host에서 먼저 처리한뒤 Kernel로 보내는 방법이 있다.
>>
>>    Compile time 에 조건을 계산해서 처리하므로 Kernel 안 분기문의 조건을 일일히 계산하는 것보다
>>
>>    빨라질 수 있다.
>>
>>      ```cpp
>>      // Example of Answer 1
>>      __global__
>>      void foo(int* a, bool cond) {
>>          if (cond) do_something()
>>          else do_something_else()
>>      }
>>      __host__
>>      bool cond = check_stuff();
>>      foo(data, cond);
>>      ```
>
>> 2. Control flow가 분기하는 대신, Data가 분기하도록 연산을 처리한다.
>>
>>      ```cpp
>>      void 
>>      // Example of Answer 2
>>      void foo(int* a, int* b) {
>>          // 여가서 check() 는 boolean value를 리턴한다.
>>          if (check(a[index]) {b[index]++;}
>>      }
>>      ```
>>      2.1. 엘비스 연산자 분기문을 이용한다.
>>
>>      ```cpp
>>      void foo(int* a, int b) {
>>          b[index] = check(a[index]) ? 1 : 0;
>>      }
>>      ```
>   [추가]


04.마침
---
여기까지가 CUDA C를 사용하기 위한 가장 기초적인 부분들. 자세한 것은 [Cuda documentation](https://docs.nvidia.com/cuda/index.html) 참고
