#pragma once
#include "Renderer.h"

class PixelBuffer
{
private:
	unsigned int m_RendererID;
	int m_Width;
	int m_Height;
public:
	PixelBuffer(const void* data, int width, int height, int bit);
	~PixelBuffer();

	void ReadPixels();

	void PackBind();
	void UnpackBind();
	void PackUnbind();
	void UnpackUnbind();

	int GetWidth() const { return m_Width; }
	int GetHeight() const { return m_Height; }
	unsigned int GetID() const { return m_RendererID; }
};