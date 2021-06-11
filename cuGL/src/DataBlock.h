#pragma once
#include "Matrix.h"
#include "cudacheck.h"

struct DataBlock
{
	Matrix* matrix;
	float* devicePointer;
	
	DataBlock(Matrix& m);
	~DataBlock();
	
	void Initialize();

	

};
