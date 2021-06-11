#pragma once
#include <string>
#include <unordered_map>


struct ShaderProgramSource
{
	std::string VertexSource;
	std::string FragmentSource;
};



class Shader
{
private:
	unsigned int m_RendererID;
	std::string FilePath;
	std::unordered_map<std::string, int> m_UniformLocationCache;
public:
	Shader(const std::string& filepath);
	~Shader();
	
	void Bind() const;
	void Unbind() const;
	void SetUniform4f(const std::string& name, float v0, float v1, float v2, float v3);
	void SetUniform1i(const std::string& name, int value);
private:
	unsigned int CreateShader(std::string vertexshader, std::string fragmentshader) const;
	unsigned int CompileShader(unsigned int type, std::string source) const;
	ShaderProgramSource ParseShader(const std::string& filepath) const;
	int GetUniformLocation(const std::string& name);
};

