#include <cuda_runtime.h>
#include <stdio.h>
#include "cudacheck.h"

__constant__ int ConstMem[2];

__global__ void SimulatorUpdateMagneticFieldsKernel(float* hz, float* ex, float* ey,
    float* Chzh, float* Chzex, float* Chzey)
{
    int indx = threadIdx.x + blockIdx.x * blockDim.x;
    int indy = threadIdx.y + blockIdx.y * blockDim.y;
    
   
    int offset = (indx + 16) + ConstMem[0] * (indy + 16);
    int offsetC = indx + ConstMem[1] * indy;
    int offsetyp = offset + ConstMem[0];
    int offsetxp = offset + 1;

    hz[offset] = Chzh[offsetC] * hz[offset] 
        + Chzex[offsetC] * (ex[offsetyp] - ex[offset]) 
        + Chzey[offsetC] * (ey[offsetxp] - ey[offset]);

}

__global__ void SimulatorUpdateElectricFieldsKernel(float* hz, float* ex, float* ey,
    float* Cexe, float* Cexhz, float* Ceye, float* Ceyhz)
{
    int indx = threadIdx.x + blockIdx.x * blockDim.x;
    int indy = threadIdx.y + blockIdx.y * blockDim.y;
    
    int offset = (indx + 16) + ConstMem[0] * (indy + 16);
    int offsetC = indx + ConstMem[1] * indy;
    int offsetyn = offset - ConstMem[0];
    int offsetxn = offset - 1;


   
    ex[offset] = Cexe[offsetC] * ex[offset]
        + Cexhz[offsetC] * (hz[offset] - hz[offsetyn]);
    ey[offset] = Ceye[offsetC] * ey[offset]
        + Ceyhz[offsetC] * (hz[offset] - hz[offsetxn]);

}

__global__ void Check(float* data, int Mconst)
{
    int indx = threadIdx.x + blockIdx.x * blockDim.x;
    int indy = threadIdx.y + blockIdx.y * blockDim.y;
    int offsetC = indx + Mconst * indy;
    if (data[offsetC] != 1.0f)
        printf("Error! ThreadId is %d , %d\n value is %f", indx, indy, data[offsetC]);
}


void SimulatorUpdateMagneticFieldsInterface(float* Hz, float* Ex, float* Ey,
    float* Chzh, float* Chzex, float* Chzey,
	dim3 Grids, dim3 Threads)
{

    SimulatorUpdateMagneticFieldsKernel << <Grids, Threads >> >
		(Hz, Ex, Ey, Chzh, Chzex, Chzey);
}

void SimulatorUpdateElectricFieldsInterface(float* Hz, float* Ex, float* Ey,
    float* Cexe, float* Cexhz, float* Ceye, float* Ceyhz,
    dim3 Grids, dim3 Threads)
{

    SimulatorUpdateElectricFieldsKernel << <Grids, Threads >> >
        (Hz, Ex, Ey, Cexe, Cexhz, Ceye, Ceyhz);
}


void SimulatorSetConstInterface(int* src, int num)
{
    static bool Done = false;
    if (!Done)
    {
        CUDACheck(cudaMemcpyToSymbol(ConstMem, src, num * sizeof(int)));
        Done = true;
    }
}


