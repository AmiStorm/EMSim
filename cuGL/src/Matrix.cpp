#include "Matrix.h"

Matrix::Matrix()
	:m_Pointer(nullptr), m_Row(0), m_Column(0),
	m_Count(0), m_Size(0)
{
	//
}

Matrix::Matrix(int M, int N)
	:m_Pointer(nullptr), m_Row(M), m_Column(N),
	m_Count(M * N), m_Size(M * N * sizeof(float))
{
	m_Pointer = new float[m_Count]();
}



Matrix::~Matrix()
{
	if (m_Pointer) delete[] m_Pointer;
}






int Matrix::GetOffset(int m, int n) const
{
	return (m - 1) + (n - 1) * m_Row;
}

float& Matrix::GetValue(int m, int n)
{
	return m_Pointer[GetOffset(m, n)];
}

float& Matrix::operator()(int m, int n)
{
	return GetValue(m, n);
}



float& Matrix::operator()(int offset1)
{
	return m_Pointer[offset1 - 1];
}

float& Matrix::operator[](int offset0)
{
	return m_Pointer[offset0];
}

void Matrix::SetValue(float value)
{
	for (int i = 0; i < m_Count; i++)
		m_Pointer[i] = value;
}

void Matrix::Circle(int CenterM, int CenterN, int radius, float value)
{
	int radius2 = radius * radius;
	for (int j = 1; j < this->GetN(); j++)
	{
		for (int i = 1; i < this->GetM(); i++)
		{
			if (GetDistance2(i, j, CenterM, CenterN) <= radius2)
				(*this)(i, j) = value;
		}
	}
}

int Matrix::GetDistance2(int M, int N, int CenterM, int CenterN)
{
	int dis2 = (M - CenterM) * (M - CenterM) + (N - CenterN) * (N - CenterN);
	return dis2;
}

