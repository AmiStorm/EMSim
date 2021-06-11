#pragma once
#include "DataBlock.h"

class Simulator
{
private:
	float* m_Hz, *m_Ex, *m_Ey;
	float* m_Chzh, *m_Chzex, *m_Chzey;
	float* m_Cexe, *m_Cexhz;
	float* m_Ceye, *m_Ceyhz;
	int m_M, m_N;
	int m_Mconst, m_Nconst;
	dim3 m_Grids, m_Threads;
public:
	Simulator(DataBlock& D_Hz, DataBlock& D_Ex, DataBlock& D_Ey, 
		DataBlock& D_Chzh, DataBlock& D_Chzex, DataBlock& D_Chzey,
		DataBlock& D_Cexe, DataBlock& D_Cexhz,
		DataBlock& D_Ceye, DataBlock& D_Ceyhz);
	~Simulator();

	void UpdateMagneticFields();
	void UpdateElectricFields();
	void SetConst(int* src, int num);
	void Update();
	void Sync();

};
