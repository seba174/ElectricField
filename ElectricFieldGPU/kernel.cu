#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SFML/Graphics.hpp>
#include <stdio.h>
#include <iostream>


__global__ void testFieldCalculation(float* pixels, unsigned int pixelsSize, int* x, int* y, unsigned int count, unsigned int constK, unsigned int width, unsigned int height)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int w = index % width;
	int h = index / width;
	//int stride = blockDim.x * gridDim.x;

	if (index < pixelsSize)
	{
		float forceX = 0, forceY = 0;
		for (int i = 0; i < count; ++i)
		{
			int dx = w - x[i];
			int dy = h - y[i];
			float magnitude = sqrtf(dx*dx + dy * dy) + 0.01;
			float chargefm = constK / magnitude;

			float xUnit = dx / magnitude;
			float yUnit = dy / magnitude;

			forceX += xUnit * chargefm;
			forceY += yUnit * chargefm;
		}
		pixels[index] = sqrtf(forceX*forceX + forceY * forceY);
	}
}

void FreeDeviceMemory(float* forceArray, int* x, int* y)
{
	cudaFree(forceArray);
	cudaFree(x);
	cudaFree(y);
}

void FreeHostMemory(float* forceArray, int* x, int* y)
{
	delete[](forceArray);
	delete[](x);
	delete[](y);
}

int main()
{
	srand(100);

	int width = 1200;
	int height = 800;
	int count = 100;

	int constK = 100;
	int pixelArraySize = width * height;

	float* forceInPixels = new float[pixelArraySize];
	int* x = new int[count];
	int* y = new int[count];

	for (int i = 0; i < count; i++)
	{
		x[i] = rand() % width;
		y[i] = rand() % height;
	}

	sf::RenderWindow window;
	window.create(sf::VideoMode(width, height), "", sf::Style::Default);

	sf::Image image;
	sf::Texture texture;
	sf::Sprite sprite;
	image.create(width, height);

	float *dev_pixels = 0;
	int *dev_x = 0;
	int *dev_y = 0;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	cudaStatus = cudaMalloc((void**)&dev_pixels, pixelArraySize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		FreeDeviceMemory(dev_pixels, dev_x, dev_y);
		FreeHostMemory(forceInPixels, x, y);
		return 1;
	}

	cudaStatus = cudaMalloc((void**)&dev_x, width * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		FreeDeviceMemory(dev_pixels, dev_x, dev_y);
		FreeHostMemory(forceInPixels, x, y);
		return 1;
	}

	cudaStatus = cudaMalloc((void**)&dev_y, height * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		FreeDeviceMemory(dev_pixels, dev_x, dev_y);
		FreeHostMemory(forceInPixels, x, y);
		return 1;
	}

	cudaStatus = cudaMemcpy(dev_x, x, width * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeDeviceMemory(dev_pixels, dev_x, dev_y);
		FreeHostMemory(forceInPixels, x, y);
		return 1;
	}

	cudaStatus = cudaMemcpy(dev_y, y, height * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		FreeDeviceMemory(dev_pixels, dev_x, dev_y);
		FreeHostMemory(forceInPixels, x, y);
		return 1;
	}

	int blockSize = 1024;
	int numBlock = ((pixelArraySize + blockSize - 1) / blockSize);

	sf::Clock clock;
	int fps = 0;
	while (window.isOpen())
	{
		if (clock.getElapsedTime().asSeconds() >= 1)
		{
			//std::cout << fps << std::endl;
			fps = 0;
			clock.restart();
		}

		testFieldCalculation <<<numBlock, blockSize >>> (dev_pixels, pixelArraySize, dev_x, dev_y, count, constK, width, height);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			FreeDeviceMemory(dev_pixels, dev_x, dev_y);
			FreeHostMemory(forceInPixels, x, y);
			return 1;
		}

		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
			FreeDeviceMemory(dev_pixels, dev_x, dev_y);
			FreeHostMemory(forceInPixels, x, y);
			return 1;
		}

		cudaStatus = cudaMemcpy(forceInPixels, dev_pixels, pixelArraySize * sizeof(float), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			FreeDeviceMemory(dev_pixels, dev_x, dev_y);
			FreeHostMemory(forceInPixels, x, y);
			return 1;
		}

		float maxForce = 0;
		for (int i = 0; i < pixelArraySize; ++i)
		{
			if (forceInPixels[i] > maxForce)
				maxForce = forceInPixels[i];
		}

		int differentColors = 1149;
		float toppercent = 0.5;
		float botpercent = 0;
		float top = (1 - toppercent)*maxForce;
		float bot = botpercent * maxForce;

		float divider = (top - bot) / differentColors;

		for (int i = 0; i < width; ++i)
		{
			for (int j = 0; j < height; ++j)
			{
				float level = (forceInPixels[i + j * width] - bot) / divider;

				if (level <= 0)
					image.setPixel(i, j, sf::Color(0, 0, 0));
				else if (level <= 255)
					image.setPixel(i, j, sf::Color(0, 0, level));
				else if (level <= 510)
					image.setPixel(i, j, sf::Color(0, level - 255, 255));
				else if (level <= 765)
					image.setPixel(i, j, sf::Color(level - 510, 255, 765 - level));
				else if (level <= 1020)
					image.setPixel(i, j, sf::Color(255, 1020 - level, 0));
				else if (level <= 1149)
					image.setPixel(i, j, sf::Color(1276 - level, 0, 0));
				else
					image.setPixel(i, j, sf::Color(127, 0, 0));
			}
		}

		texture.loadFromImage(image);
		sprite.setTexture(texture);

		window.clear();
		window.draw(sprite);
		window.display();
		fps++;
	}

	FreeDeviceMemory(dev_pixels, dev_x, dev_y);
	FreeHostMemory(forceInPixels, x, y);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}


