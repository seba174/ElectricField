#pragma once
#include "ChargesManager.h"


class GPUElectricFieldCalculator
{
private:
	ChargesManager& chargesManager;
	float* electricFieldValues_device;
	int* xCoordinates_device;
	int* yCoordinates_device;
	int baseElectricForceMultiplier;
	int maxChargesInOneThreadRun;
	int width;
	int height;
	int blockSize;

public:

	GPUElectricFieldCalculator(ChargesManager& chargesManager, int blockSize, int baseElectricForceMultiplier);

	bool SetDevice(int id);

	bool CreateDeviceElectricFieldMatrix(int width, int height);

	bool CreateDeviceChargesArrays();

	bool UpdateDeviceChargesArrays();

	bool StartCalculatingElectricField(int maxChargesInOneThreadRun);

	bool SynchronizeDeviceAndCopyResult(float* electricFieldMatrix);

	~GPUElectricFieldCalculator();
};
