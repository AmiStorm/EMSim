#include "DataBlock.h"

DataBlock::DataBlock(Matrix& m)
	:matrix(&m), devicePointer(nullptr)
{
	CUDACheck(cudaMalloc(&devicePointer, matrix->GetSize()));
}
void DataBlock::Initialize()
{
	CUDACheck(cudaMemcpy(devicePointer, matrix->GetPointer(), matrix->GetSize(), cudaMemcpyHostToDevice));
}

DataBlock::~DataBlock()
{
	if (devicePointer)
		cudaFree(devicePointer);
}