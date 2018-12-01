#pragma once
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ElectricFieldCalculatorKernel.h"

__global__ void StartFieldCalculation(float* pixels, int pixelsSize, int* x, int* y, int count, int maxInStep, int constK, int width)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int str = index / pixelsSize;
	index = index % pixelsSize;

	int w = index % width;
	int h = index / width;
	
	int dx, dy;
	float magnitude, chargefm, xUnit, yUnit, forceX = 0, forceY = 0;
	int loopEnd = (str + 1) * maxInStep;

	for (int i = str * maxInStep; i < loopEnd && i < count; ++i)
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
	atomicAdd(&(pixels[index]), sqrtf(forceX * forceX + forceY * forceY));
	//pixels[index] += sqrtf(forceX * forceX + forceY * forceY);
}

void LaunchStartFieldCalculation(float* pixels, int pixelsSize, int* x, int* y, int count, int maxInStep, int constK, int width, int numberOfBlocks, int blockSize)
{
	StartFieldCalculation <<< numberOfBlocks, blockSize >>> (pixels, pixelsSize, x, y, count, maxInStep, constK, width);
}
