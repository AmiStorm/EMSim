#pragma once
#include <iostream>
#include <cuda_runtime.h>
#define CUDACheck(x) CUDALOGCall(x, #x, __FILE__, __LINE__)

static void CUDALOGCall(cudaError_t err,
    const char* function, const char* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cout << "[CUDA Error] (" << cudaGetErrorString(err) << "):" << function <<
            " " << file << ":" << line << std::endl;
        __debugbreak();
    }

}

class GPUTimer
{
private:
    cudaEvent_t start, end;
    float* time;
public:
    GPUTimer(float& outtime)
        :time(&outtime)
    {
       CUDACheck(cudaEventCreate(&start));
       CUDACheck(cudaEventCreate(&end));
       CUDACheck(cudaEventRecord(start));
    }

    
    ~GPUTimer()
    {
        CUDACheck(cudaEventRecord(end));
        CUDACheck(cudaEventSynchronize(end));
        
        CUDACheck(cudaEventElapsedTime(time, start, end));

        CUDACheck(cudaEventDestroy(start));
        CUDACheck(cudaEventDestroy(end));
    }

    
    
};