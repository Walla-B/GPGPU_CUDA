#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __cplusplus
extern "C" {
#endif

	class CudaTest {
	public:
		CudaTest(void);
		virtual ~CudaTest(void);
		int sum_cuda(int a, int b, int* c);

	};

#ifdef __cplusplus
}
#endif // __cplusplus
