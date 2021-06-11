#pragma once

#include "PixelBuffer.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>




class MyResource
{
private:
	PixelBuffer* m_PixelBuffer;
	cudaGraphicsResource* m_Resource;
	uchar4* m_Pointer;
	size_t m_Size;
	bool ifGeted = false;
public:
	MyResource(PixelBuffer& pb);
	~MyResource();

	void Map();
	void UnMap();
	uchar4* GetPointer() const { return m_Pointer; };
	size_t GetSize() const { return m_Size; };

	
};
