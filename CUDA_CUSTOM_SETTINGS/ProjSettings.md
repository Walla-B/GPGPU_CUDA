VStudio 2019 + CUDA 10.1 Project 세팅 방법
===
  [CUDA 10.1 설치](https://developer.nvidia.com/cuda-10.1-download-archive-base)

프로젝트 세팅방법
---
   + 템플릿에서  CUDA 프로젝트 만들기
   + 빈 프로젝트에서 만들기



1. CUDA 템플릿 이용, 프로젝트 만들기
----
    새 프로젝트 만들기 > CUDA 10.1 Runtime

2. 빈 프로젝트에서 만들기
----
    프로젝트 RMB >
    
        Build dependencies > build customization > CUDA 10.1 체크

        Properties > Configuration >

            Configuration & Platform 이 컴파일 설정과 동일해야 함

            > CUDA C/C++ > Common > Target Machine platform > 위의 Platform과 동일하게

            > Linker > Input > Additional dependencies > cuda.lib cudart.lib 추가

    이후, .cu 파일에 #include "cuda_runtime.h" 추가
