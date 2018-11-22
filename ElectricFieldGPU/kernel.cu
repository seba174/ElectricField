#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SFML/Graphics.hpp>
#include <stdio.h>
#include <iostream>

#include "INI_Reader.h"
#include "ChargesManager.h"
#include "GPUElectricFieldCalculator.h"


__global__ void testFieldCalculation(float* pixels, int pixelsSize, int* x, int* y, int count, int constK, int width)
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
	srand(300);

	const std::string baseConfigGroup = "Config";
	INI_Reader config("config.ini");

	int minSpeed = 1;
	int maxSpeed = 5;
	int width = 640;
	int height = 480;
	int count = 1024;
	int constK = 100;
	float toppercent = 0.1;
	float botpercent = 0;
	int blockSize = 1024;

	try
	{
		width = std::stoi(config.getValue(baseConfigGroup, "Width"));
		height = std::stoi(config.getValue(baseConfigGroup, "Height"));
		count = std::stoi(config.getValue(baseConfigGroup, "NumberOfCharges"));
		constK = std::stoi(config.getValue(baseConfigGroup, "ConstK"));
		toppercent = std::stof(config.getValue(baseConfigGroup, "TopPercent"));
		botpercent = std::stof(config.getValue(baseConfigGroup, "BotPercent"));
	}
	catch (std::invalid_argument exception) { }

	int pixelArraySize = width * height;
	float* forceInPixels = new float[pixelArraySize];
	int* x = new int[count];
	int* y = new int[count];
	int* velocityX = new int[count];
	int* velocityY = new int[count];

	for (int i = 0; i < count; i++)
	{
		x[i] = rand() % width;
		y[i] = rand() % height;
		velocityX[i] = rand() % (maxSpeed - minSpeed) + minSpeed;
		velocityY[i] = rand() % (maxSpeed - minSpeed) + minSpeed;
		if (rand() % 2 == 0)
			velocityX[i] = -velocityX[i];
		if (rand() % 2 == 0)
			velocityY[i] = -velocityY[i];
	}

	ChargesManager chargesManager(count);
	GPUElectricFieldCalculator gpuElectricFieldCalculator(chargesManager, blockSize);

	gpuElectricFieldCalculator.SetDevice(0);
	gpuElectricFieldCalculator.CreateDeviceElectricFieldMatrix(width, height);
	gpuElectricFieldCalculator.CreateDeviceChargesArrays();


	sf::RenderWindow window;
	window.create(sf::VideoMode(width, height), "", sf::Style::Default);
	window.setVerticalSyncEnabled(true);

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
		FreeDeviceMemory(dev_pixels, dev_x, dev_y);
		FreeHostMemory(forceInPixels, x, y);
		return 1;
	}

	cudaStatus = cudaMalloc((void**)&dev_pixels, pixelArraySize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		FreeDeviceMemory(dev_pixels, dev_x, dev_y);
		FreeHostMemory(forceInPixels, x, y);
		return 1;
	}

	cudaStatus = cudaMalloc((void**)&dev_x, count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		FreeDeviceMemory(dev_pixels, dev_x, dev_y);
		FreeHostMemory(forceInPixels, x, y);
		return 1;
	}

	cudaStatus = cudaMalloc((void**)&dev_y, count * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		FreeDeviceMemory(dev_pixels, dev_x, dev_y);
		FreeHostMemory(forceInPixels, x, y);
		return 1;
	}




	int numBlock = ((pixelArraySize + blockSize - 1) / blockSize);

	float maxForce = 15 * constK;

	sf::Clock clock;
	int fps = 0;
	while (window.isOpen())
	{
		//{ cp
		cudaStatus = cudaMemcpy(dev_x, x, count * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			FreeDeviceMemory(dev_pixels, dev_x, dev_y);
			FreeHostMemory(forceInPixels, x, y);
			return 1;
		}

		cudaStatus = cudaMemcpy(dev_y, y, count * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			FreeDeviceMemory(dev_pixels, dev_x, dev_y);
			FreeHostMemory(forceInPixels, x, y);
			return 1;
		}

		//gpuElectricFieldCalculator.UpdateDeviceChargesArrays();

		if (clock.getElapsedTime().asSeconds() >= 25)
		{
			std::cout << fps << std::endl;
			fps = 0;
			clock.restart();
		}

		//gpuElectricFieldCalculator.StartCalculatingElectricField();

		testFieldCalculation <<< numBlock, blockSize >>> (dev_pixels, pixelArraySize, dev_x, dev_y, count, constK, width);

		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		for (int i = 0; i < count; i++)
		{
			if (x[i] + velocityX[i] < 0)
			{
				x[i] = 0;
				velocityX[i] = -velocityX[i];
			}
			else if (x[i] + velocityX[i] >= width)
			{
				x[i] = width - 1;
				velocityX[i] = -velocityX[i];
			}
			else
			{
				x[i] += velocityX[i];
			}

			if (y[i] + velocityY[i] < 0)
			{
				y[i] = 0;
				velocityY[i] = -velocityY[i];
			}
			else if (y[i] + velocityY[i] >= height)
			{
				y[i] = height - 1;
				velocityY[i] = -velocityY[i];
			}
			else
			{
				y[i] += velocityY[i];
			}
		}

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			FreeDeviceMemory(dev_pixels, dev_x, dev_y);
			FreeHostMemory(forceInPixels, x, y);
			return 1;
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

		//float maxForce = 0;
		//for (int i = 0; i < pixelArraySize; ++i)
		//{
		//	if (forceInPixels[i] > maxForce)
		//		maxForce = forceInPixels[i];
		//}
		//maxForce = 1000;

		int differentColors = 1149;
		float top = (1 - toppercent) * maxForce;
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


