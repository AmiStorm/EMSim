#include "MyResource.h"
#include "cudacheck.h"

MyResource::MyResource(PixelBuffer& pb)
	:m_PixelBuffer(&pb), m_Pointer(nullptr), m_Size(0)
{
	CUDACheck(cudaGraphicsGLRegisterBuffer(&m_Resource, m_PixelBuffer->GetID(), cudaGraphicsMapFlagsNone));
}

MyResource::~MyResource()
{
	m_Pointer = nullptr;
	m_Size = 0;
}

void MyResource::Map()
{
	CUDACheck(cudaGraphicsMapResources(1, &m_Resource, 0));
	if (!ifGeted)
	{
		CUDACheck(cudaGraphicsResourceGetMappedPointer((void**)&m_Pointer, &m_Size, m_Resource));
		ifGeted = true;
	}
}

void MyResource::UnMap()
{
	CUDACheck(cudaGraphicsUnmapResources(1, &m_Resource, 0));
}

