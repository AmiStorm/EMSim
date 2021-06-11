#include "Simulator.h"
#define TILE_SIZE 16


void SimulatorUpdateMagneticFieldsInterface(float* Hz, float* Ex, float* Ey,
	float* Chzh, float* Chzex, float* Chzey,
	dim3 Grids, dim3 Threads);
void SimulatorUpdateElectricFieldsInterface(float* Hz, float* Ex, float* Ey,
	float* Cexe, float* Cexhz, float* Ceye, float* Ceyhz,
	dim3 Grids, dim3 Threads);
void SimulatorSetConstInterface(int* src, int num);

Simulator::Simulator(DataBlock& D_Hz, DataBlock& D_Ex, DataBlock& D_Ey,
	DataBlock& D_Chzh, DataBlock& D_Chzex, DataBlock& D_Chzey,
	DataBlock& D_Cexe, DataBlock& D_Cexhz,
	DataBlock& D_Ceye, DataBlock& D_Ceyhz)
	:m_Hz(D_Hz.devicePointer), m_Ex(D_Ex.devicePointer), m_Ey(D_Ey.devicePointer),
	m_Chzh(D_Chzh.devicePointer), m_Chzex(D_Chzex.devicePointer), m_Chzey(D_Chzey.devicePointer), 
	m_Cexe(D_Cexe.devicePointer), m_Cexhz(D_Cexhz.devicePointer), 
	m_Ceye(D_Ceye.devicePointer), m_Ceyhz(D_Ceyhz.devicePointer), 
	m_M(0), m_N(0),
	m_Mconst(0), m_Nconst(0),
	m_Grids(0), m_Threads(0)
{
	if ( (D_Hz.matrix->GetM() == D_Ex.matrix->GetM())
		&& 
		(D_Hz.matrix->GetM() == D_Ey.matrix->GetM()) )
	{
		m_M = D_Hz.matrix->GetM();
	}
	if ((D_Hz.matrix->GetN() == D_Ex.matrix->GetN())
		&&
		(D_Hz.matrix->GetN() == D_Ey.matrix->GetN()))
	{
		m_N = D_Hz.matrix->GetN();
	}
	if (m_M % TILE_SIZE)
	{
		std::cout << "Rows of Fields is wrong!" << std::endl;
		__debugbreak();
	}
	else
	{
		m_Grids.x = (m_M / TILE_SIZE) - 2;
	}
	if (m_N % TILE_SIZE)
	{
		std::cout << "Columns of Fields is wrong!" << std::endl;
		__debugbreak();
	}
	else
	{
		m_Grids.y = (m_N / TILE_SIZE) - 2;
	}
	m_Grids.z = 1;
	m_Threads = dim3(TILE_SIZE, TILE_SIZE);

	m_Mconst = D_Chzh.matrix->GetM();
	m_Nconst = D_Chzh.matrix->GetN();

	if (m_Mconst % TILE_SIZE || (m_M - m_Mconst != 32))
	{
		std::cout << "Rows of Const is wrong!" << std::endl;
		__debugbreak();
	}

	if (m_Nconst % TILE_SIZE || (m_N - m_Nconst != 32))
	{
		std::cout << "Columns of Const is wrong!" << std::endl;
		__debugbreak();
	}
	

}

Simulator::~Simulator()
{
}

void Simulator::UpdateMagneticFields()
{
	SimulatorUpdateMagneticFieldsInterface(m_Hz, m_Ex, m_Ey,
		m_Chzh, m_Chzex, m_Chzey,
		m_Grids, m_Threads);
}

void Simulator::UpdateElectricFields()
{
	SimulatorUpdateElectricFieldsInterface(m_Hz, m_Ex, m_Ey,
		m_Cexe, m_Cexhz, m_Ceye, m_Ceyhz,
		m_Grids, m_Threads);
}

void Simulator::SetConst(int* src, int num)
{
	SimulatorSetConstInterface(src, num);
}

void Simulator::Update()
{
	UpdateMagneticFields();
	UpdateElectricFields();
}

void Simulator::Sync()
{
	cudaDeviceSynchronize();
}
