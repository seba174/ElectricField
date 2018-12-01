#include "CPUElectricFieldCalculator.h"
#include <cmath>


CPUElectricFieldCalculator::CPUElectricFieldCalculator(const ChargesManager& chargesManager, int width, int height, int baseElectricForceMultiplier)
	:chargesManager(chargesManager)
{
	UpdateParameters(width, height, baseElectricForceMultiplier);
}

void CPUElectricFieldCalculator::UpdateParameters(int width, int height, int baseElectricForceMultiplier)
{
	this->width = width;
	this->height = height;
	this->baseElectricForceMultiplier = baseElectricForceMultiplier;
}

void CPUElectricFieldCalculator::CalculateElectricField(float * electricFieldMatrix)
{
	int* x = chargesManager.GetXCoordinates();
	int* y = chargesManager.GetYCoordinates();
	for (int dim = 0; dim < width * height; ++dim)
	{
		int w = dim % width;
		int h = dim / width;
		float forceX = 0, forceY = 0;
		for (int k = 0; k < chargesManager.GetNumberOfCharges(); ++k)
		{
			int dx = w - x[k];
			int dy = h - y[k];
			float magnitude = sqrtf(static_cast<float>(dx * dx + dy * dy)) + 0.01f;
			float chargefm = baseElectricForceMultiplier / magnitude;

			float xUnit = dx / magnitude;
			float yUnit = dy / magnitude;

			forceX += abs(xUnit * chargefm);
			forceY += abs(yUnit * chargefm);
		}

		electricFieldMatrix[dim] = sqrtf(forceX * forceX + forceY * forceY);
	}
}
