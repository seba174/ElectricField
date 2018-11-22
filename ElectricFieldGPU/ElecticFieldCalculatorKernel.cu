#pragma once
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ElectricFieldCalculatorKernel.h"

__global__ void StartFieldCalculation(float* pixels, int pixelsSize, int* x, int* y, int count, int constK, int width)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int w = index % width;
	int h = index / width;

	if (index < pixelsSize)
	{
		int dx, dy;
		float magnitude, chargefm, xUnit, yUnit, forceX = 0, forceY = 0;

		for (int i = 0; i < count; ++i)
		{
			dx = x[i] - w;
			dy = y[i] - h;

			magnitude = sqrtf(dx * dx + dy * dy) + 0.01f;
			chargefm = constK / magnitude;

			xUnit = dx / magnitude;
			yUnit = dy / magnitude;

			forceX += abs(xUnit * chargefm);
			forceY += abs(yUnit * chargefm);
		}
		pixels[index] = sqrtf(forceX * forceX + forceY * forceY);
	}
}

void LaunchStartFieldCalculation(float* pixels, int pixelsSize, int* x, int* y, int count, int constK, int width, int numberOfBlocks, int blockSize)
{
	StartFieldCalculation <<< numberOfBlocks, blockSize >>> (pixels, pixelsSize, x, y, count, constK, width);
}
