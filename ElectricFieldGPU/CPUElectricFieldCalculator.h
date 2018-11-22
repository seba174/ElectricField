#pragma once
#include "ChargesManager.h"


class CPUElectricFieldCalculator
{
private:
	const ChargesManager& chargesManager;
	int baseElectricForceMultiplier;
	int width;
	int height;

public:
	CPUElectricFieldCalculator(const ChargesManager& chargesManager, int width, int height, int baseElectricForceMultiplier);

	void UpdateParameters(int width, int height, int baseElectricForceMultiplier);

	void CalculateElectricField(float* electricFieldMatrix);
};