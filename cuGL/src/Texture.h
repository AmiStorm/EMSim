#pragma once
#include "Renderer.h"
class PixelBuffer;

class Texture
{
private:
	unsigned int m_RendererID;
	std::string m_FilePath;
	unsigned char* m_LocalBuffer;
	int m_Width, m_Height, m_BPP;
public:
	Texture(const std::string& filepath);
	Texture();
	
	~Texture();

	void Bind(unsigned int slot = 0) const;
	void Unbind() const;

	void Read(PixelBuffer& pixelbuffer, unsigned int slot);

	void ReadRGBA(PixelBuffer& pixelbuffer, unsigned int slot);

	inline int GetWidth() const { return m_Width; }
	inline int GetHeight() const { return m_Height; }
	inline int GetBPP() const { return m_BPP; }

	
};