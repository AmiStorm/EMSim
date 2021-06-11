#include "Detector.h"

Detector::Detector(DataBlock& D_Fields, Translator& camera, Translator& world)
    :m_Camera(camera), m_World(world), m_Fields(D_Fields.devicePointer),
    m_Intensity(nullptr),
    m_PixelBuffer(nullptr, m_Camera.M, m_Camera.N, 4),
    m_R(m_PixelBuffer),
    m_DetectGrids(m_Camera.M / 16, m_Camera.N / 16), m_DetectThreads(16, 16),
    m_DrawGrids(m_Camera.M / 16, m_Camera.N / 16), m_DrawThreads(16, 16),
    m_ispower(true)
{
    SetIntensity();
    m_DetectGrids = dim3(m_Camera.M / 16, m_Camera.N / 16);
    m_DetectThreads = dim3(16, 16);
    m_DrawGrids = dim3(m_Camera.M / 16, m_Camera.N / 16);
    m_DrawThreads = dim3(16, 16);
}


Detector::~Detector()
{
    cudaFree(m_Intensity);
}
void Detector::Detect()
{
    if (m_ispower)
    {
        DetectorDetectInterface(m_Fields, m_Intensity, m_World.M, m_Camera.M,
            m_DetectGrids, m_DetectThreads, m_ispower);
    }
    else 
    {
        DetectorDetectInterface(m_Fields, m_Intensity, m_World.M, m_Camera.M,
            m_DetectGrids, m_DetectThreads, m_ispower);
    }

}

void Detector::DrawPixel(float contrust)
{
    m_R.Map();
    if (m_ispower)
    {
        DetectorDrawPixelInterface(m_R.GetPointer(), m_Intensity, contrust,
            m_DrawGrids, m_DrawThreads, m_ispower);
    }
    else
    {
        DetectorDrawPixelInterface(m_R.GetPointer(), m_Intensity, contrust,
            m_DrawGrids, m_DrawThreads, m_ispower);
    }
    m_R.UnMap();
}

void Detector::Sync() const
{
    cudaDeviceSynchronize();
}

void Detector::SetPowerMode(bool ispower)
{
    m_ispower = ispower;
}

void Detector::SetIntensity()
{
    int size = m_Camera.M * m_Camera.N * sizeof(float);
    CUDACheck(cudaMalloc(&m_Intensity, size));
    CUDACheck(cudaMemset(m_Intensity, 0, size));
}

