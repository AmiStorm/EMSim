#pragma once


class Matrix
{
private:
	float* m_Pointer;
	int m_Row;
	int m_Column;
	int m_Count;
	int m_Size;


public:
	Matrix();
	Matrix(int M, int N);
	~Matrix();

	float* GetPointer() const { return m_Pointer; }
	int GetM() const { return m_Row; }
	int GetN() const { return m_Column; }
	int GetCount() const { return m_Count; }
	int GetSize() const { return m_Size; }

	int GetOffset(int m, int n) const;
	float& GetValue(int m, int n);
	float& operator()(int m, int n);
	float& operator()(int offset1);
	float& operator[](int offset0);

	void SetValue(float value);
	void Circle(int CenterM, int CenterN, int radius, float value);


private:
	int GetDistance2(int M, int N, int CenterM, int CenterN);
};
