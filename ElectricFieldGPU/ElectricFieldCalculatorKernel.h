#pragma once


extern "C" void  LaunchStartFieldCalculation(float* pixels, int pixelsSize, int* x, int* y,
	int count, int constK, int width, int numberOfBlocks, int blockSize);

