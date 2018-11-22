#pragma once
#include "ChargesManager.h"


class GPUElectricFieldCalculator
{
private:
	const ChargesManager& chargesManager;
	float* electricFieldValues_device;
	int* xCoordinates_device;
	int* yCoordinates_device;
	int baseElectricForceMultiplier;
	int width;
	int height;
	int blockSize;

public:

	GPUElectricFieldCalculator(const ChargesManager& chargesManager, int blockSize, int baseElectricForceMultiplier);

	bool SetDevice(int id);

	bool CreateDeviceElectricFieldMatrix(int width, int height);

	bool CreateDeviceChargesArrays();

	bool UpdateDeviceChargesArrays();

	bool StartCalculatingElectricField();

	bool SynchronizeDeviceAndCopyResult(float* electricFieldMatrix);

	~GPUElectricFieldCalculator();
};
