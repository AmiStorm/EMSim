#include <cuda_runtime.h>
#include <helper_cuda.h>



__global__ void DetectorPowerDetectKernel(float* field, float* intensity, int M_Fields, int M_Camera)
{
    int indx = threadIdx.x + blockIdx.x * blockDim.x;
    int indy = threadIdx.y + blockIdx.y * blockDim.y;
    int offsetView = indx + M_Camera * indy;
    int offset = (indx + 16) + M_Fields * (indy + 16);
    intensity[offsetView] = field[offset] * field[offset];
}

__global__ void DetectorFieldDetectKernel(float* field, float* intensity, int M_Fields, int M_Camera)
{
    int indx = threadIdx.x + blockIdx.x * blockDim.x;
    int indy = threadIdx.y + blockIdx.y * blockDim.y;
    int offsetView = indx + M_Camera * indy;
    int offset = (indx + 16) + M_Fields * (indy + 16);
    intensity[offsetView] = field[offset];
}


__global__ void DetectorPowerDrawPixelKernel(uchar4* ptr, float* intensity, float contrust)
{

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (1)
    {

        int offset = x + y * blockDim.x * gridDim.x;

       
        uchar4 color = { 0, 255, 0, 255};

        float greenf = intensity[offset] / contrust;
        if (greenf < 1.0f)
            color.y = (unsigned char)roundf(255.0f * greenf);
        
        ptr[offset] = color;
       
    }
       
}

__global__ void DetectorFieldDrawPixelKernel(uchar4* ptr, float* intensity, float contrust)
{

    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    float k = intensity[offset]; 
    
    if (k >= 0)
    {
        uchar4 color = { 255, 0, 0, 255 };
        k = k * k / contrust;
        if (k < 1.0f)
            color.x = (unsigned char)roundf(255.0f * k);

        ptr[offset] = color;

    }
    else
    {
        uchar4 color = { 0, 0, 255, 255 };
        k = k * k / contrust;
        if (k < 1.0f)
            color.z = (unsigned char)roundf(255.0f * k);

        ptr[offset] = color;
    }
    

}



void DetectorDetectInterface(float* field, float* intensity, int M_field, int M_camera,
    dim3 grids, dim3 threads, bool ispower)
{
    if (ispower)
        DetectorPowerDetectKernel << <grids, threads >> > (field, intensity, M_field, M_camera);
    else
        DetectorFieldDetectKernel << <grids, threads >> > (field, intensity, M_field, M_camera);
}

void DetectorDrawPixelInterface(uchar4* ptr, float* intensity, float contrust,
    dim3 grids, dim3 threads, bool ispower)
{
    if (ispower)
        DetectorPowerDrawPixelKernel << <grids, threads >> > (ptr, intensity, contrust);
    else
        DetectorFieldDrawPixelKernel <<<grids, threads>>> (ptr, intensity, contrust);
}

