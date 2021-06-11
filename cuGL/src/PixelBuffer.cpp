#include "PixelBuffer.h"

PixelBuffer::PixelBuffer(const void* data, int width, int height, int bit)   
    :m_Width(width), m_Height(height)
{
    int size = width * height * bit;
    GLCall(glGenBuffers(1, &m_RendererID));
    GLCall(glBindBuffer(GL_PIXEL_PACK_BUFFER, m_RendererID));
    GLCall(glBufferData(GL_PIXEL_PACK_BUFFER, size, data, GL_STREAM_COPY));
    GLCall(glBindBuffer(GL_PIXEL_PACK_BUFFER, 0));
}

PixelBuffer::~PixelBuffer()
{
    GLCall(glDeleteBuffers(1, &m_RendererID));
}

void PixelBuffer::ReadPixels()
{
    GLCall(glBindBuffer(GL_PIXEL_PACK_BUFFER, m_RendererID));
    GLCall(glReadPixels(0, 0, m_Width, m_Height, GL_RGB, GL_UNSIGNED_BYTE, nullptr));
    GLCall(glBindBuffer(GL_PIXEL_PACK_BUFFER, 0));
}

void PixelBuffer::PackBind()
{
    GLCall(glBindBuffer(GL_PIXEL_PACK_BUFFER, m_RendererID));
}

void PixelBuffer::UnpackBind()
{
    GLCall(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_RendererID));
}

void PixelBuffer::PackUnbind()
{
    GLCall(glBindBuffer(GL_PIXEL_PACK_BUFFER, 0));
}

void PixelBuffer::UnpackUnbind()
{
    GLCall(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
}
