#pragma once
#include <SFML/Graphics.hpp>
#include <iostream>

#include "ConfigLoader.h"
#include "ChargesManager.h"
#include "GPUElectricFieldCalculator.h"
#include "ElectricFieldImageCreator.h"

const std::string configFileName = "Config.ini";
const int baseElectricForceMultiplier = 100;
const int maxForce = 15 * baseElectricForceMultiplier;

void CleanUp(float*);


int main()
{
	srand(300);
	ConfigLoader configLoader(configFileName);

	int minSpeed = configLoader.GetMinVelocity();
	int maxSpeed = configLoader.GetMaxVelocity();
	int width = configLoader.GetWidth();
	int height = configLoader.GetHeight();
	int numberOfCharges = configLoader.GetNumberOfCharges();
	float electricForceCoefficient = configLoader.GetElectricForceCoefficient();
	int blockSize = configLoader.GetBlockSize();

	float* forceInPixels = new float[width * height];

	ChargesManager chargesManager(numberOfCharges, width, height);
	GPUElectricFieldCalculator gpuElectricFieldCalculator(chargesManager, blockSize, baseElectricForceMultiplier);

	chargesManager.SetRandomPositions();
	chargesManager.SetRandomVelocities(minSpeed, maxSpeed);

	if (!gpuElectricFieldCalculator.SetDevice(0))
	{
		CleanUp(forceInPixels);
		return 1;
	}
	if (!gpuElectricFieldCalculator.CreateDeviceElectricFieldMatrix(width, height))
	{
		CleanUp(forceInPixels);
		return 1;
	}
	if (!gpuElectricFieldCalculator.CreateDeviceChargesArrays())
	{
		CleanUp(forceInPixels);
		return 1;
	}

	sf::RenderWindow window;
	window.create(sf::VideoMode(width, height), "", sf::Style::Default);
	window.setVerticalSyncEnabled(true);

	ElectricFieldImageCreator imageCreator(width, height, electricForceCoefficient, maxForce);

	sf::Event event;
	sf::Clock clock;
	int fps = 0;
	while (window.isOpen())
	{
		if (!gpuElectricFieldCalculator.UpdateDeviceChargesArrays())
		{
			CleanUp(forceInPixels);
			return 1;
		}

		if (clock.getElapsedTime().asSeconds() >= 25)
		{
			std::cout << fps << std::endl;
			fps = 0;
			clock.restart();
		}

		if (!gpuElectricFieldCalculator.StartCalculatingElectricField())
		{
			CleanUp(forceInPixels);
			return 1;
		}

		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
			else if (event.type == sf::Event::Resized)
			{
				CleanUp(forceInPixels);
				window.setView(sf::View(sf::FloatRect(0, 0, event.size.width, event.size.height)));
				width = event.size.width;
				height = event.size.height;
				chargesManager.UpdateBounds(width, height);
				forceInPixels = new float[width * height];

				if (!gpuElectricFieldCalculator.CreateDeviceElectricFieldMatrix(width, height))
				{
					CleanUp(forceInPixels);
					return 1;
				}

				imageCreator = ElectricFieldImageCreator(width, height, electricForceCoefficient, maxForce);
			}
		}

		chargesManager.UpdatePositions();

		if (!gpuElectricFieldCalculator.SynchronizeDeviceAndCopyResult(forceInPixels))
		{
			CleanUp(forceInPixels);
			return 1;
		}

		imageCreator.UpdateImage(forceInPixels);
		window.clear();
		window.draw(imageCreator.GetSprite());
		window.display();
		fps++;
	}

	return 0;
}

void CleanUp(float* arr)
{
	if (arr != nullptr)
		delete[] arr;
}