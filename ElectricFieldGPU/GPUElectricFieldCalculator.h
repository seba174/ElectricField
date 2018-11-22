#pragma once
#include "cuda_runtime.h"
#include "ChargesManager.h"


class GPUElectricFieldCalculator
{
private:
	ChargesManager chargesManager;
	float* electricFieldValues_device;
	int* xCoordinates_device;
	int* yCoordinates_device;
	int width;
	int height;
	int blockSize;

public:
	GPUElectricFieldCalculator(ChargesManager chargesManager, int blockSize);

	bool SetDevice(int id);

	bool CreateDeviceElectricFieldMatrix(int width, int height);

	bool CreateDeviceChargesArrays();

	bool UpdateDeviceChargesArrays();

	bool StartCalculatingElectricField();

	bool SynchronizeDeviceAndCopyResult(float* electricFieldMatrix);
};
