#pragma once
#include <SFML/Graphics.hpp>
#include <iostream>

#include "ConfigLoader.h"
#include "ChargesManager.h"
#include "GPUElectricFieldCalculator.h"
#include "CPUElectricFieldCalculator.h"
#include "ElectricFieldImageCreator.h"

const std::string appTitle = "ElectricField";
const std::string configFileName = "Config.ini";
const int clockTickInSeconds = 10;
const int baseElectricForceMultiplier = 100;
const int maxForce = 15 * baseElectricForceMultiplier;

void CleanUp(float*);


int main()
{
	srand(300);
	ConfigLoader config(configFileName);
	config.LoadConfig();

	float* electricFieldMatrix = new float[config.width * config.height];

	ChargesManager chargesManager(config.numberOfCharges, config.width, config.height);
	GPUElectricFieldCalculator gpuElectricFieldCalculator(chargesManager, config.blockSize, baseElectricForceMultiplier);
	CPUElectricFieldCalculator cpuElectricFieldCalculator(chargesManager, config.width, config.height, baseElectricForceMultiplier);

	chargesManager.SetRandomPositions();
	chargesManager.SetRandomVelocities(config.minSpeed, config.maxSpeed);

	if (!gpuElectricFieldCalculator.SetDevice(0))
	{
		CleanUp(electricFieldMatrix);
		return 1;
	}
	if (!gpuElectricFieldCalculator.CreateDeviceElectricFieldMatrix(config.width, config.height))
	{
		CleanUp(electricFieldMatrix);
		return 1;
	}
	if (!gpuElectricFieldCalculator.CreateDeviceChargesArrays())
	{
		CleanUp(electricFieldMatrix);
		return 1;
	}

	sf::RenderWindow window;
	window.create(sf::VideoMode(config.width, config.height), appTitle, sf::Style::Default);
	window.setVerticalSyncEnabled(true);

	ElectricFieldImageCreator imageCreator(config.width, config.height, config.electricForceCoefficient, maxForce);

	sf::Event event;
	sf::Clock clock;
	int fps = 0;
	while (window.isOpen())
	{
		if (config.isGpuModeEnabled && !gpuElectricFieldCalculator.UpdateDeviceChargesArrays())
		{
			CleanUp(electricFieldMatrix);
			return 1;
		}

		if (clock.getElapsedTime().asSeconds() >= clockTickInSeconds)
		{
			std::cout << fps << std::endl;
			fps = 0;
			clock.restart();
		}

		if (config.isGpuModeEnabled)
		{
			if (!gpuElectricFieldCalculator.StartCalculatingElectricField())
			{
				CleanUp(electricFieldMatrix);
				return 1;
			}
		}
		else
		{
			cpuElectricFieldCalculator.CalculateElectricField(electricFieldMatrix);
		}


		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
			else if (event.type == sf::Event::Resized)
			{
				CleanUp(electricFieldMatrix);
				window.setView(sf::View(sf::FloatRect(0, 0, static_cast<float>(event.size.width), static_cast<float>(event.size.height))));
				config.width = event.size.width;
				config.height = event.size.height;
				chargesManager.UpdateBounds(config.width, config.height);
				electricFieldMatrix = new float[config.width * config.height];

				if (!gpuElectricFieldCalculator.CreateDeviceElectricFieldMatrix(config.width, config.height))
				{
					CleanUp(electricFieldMatrix);
					return 1;
				}
				cpuElectricFieldCalculator.UpdateParameters(config.width, config.height, baseElectricForceMultiplier);

				imageCreator = ElectricFieldImageCreator(config.width, config.height, config.electricForceCoefficient, maxForce);
			}
			else if (event.type == sf::Event::KeyPressed && event.key.code == sf::Keyboard::R && event.key.control)
			{
				if (!gpuElectricFieldCalculator.SynchronizeDeviceAndCopyResult(electricFieldMatrix))
				{
					CleanUp(electricFieldMatrix);
					return 1;
				}

				config.LoadConfig();
				CleanUp(electricFieldMatrix);
				electricFieldMatrix = new float[config.width * config.height];
				window.setSize(sf::Vector2u(config.width, config.height));
				window.setView(sf::View(sf::FloatRect(0, 0, static_cast<float>(config.width), static_cast<float>(config.height))));
				chargesManager.UpdateSize(config.numberOfCharges, config.width, config.height);
				chargesManager.SetRandomPositions();
				chargesManager.SetRandomVelocities(config.minSpeed, config.maxSpeed);
				if (!gpuElectricFieldCalculator.CreateDeviceChargesArrays())
				{
					CleanUp(electricFieldMatrix);
					return 1;
				}
				if (!gpuElectricFieldCalculator.CreateDeviceElectricFieldMatrix(config.width, config.height))
				{
					CleanUp(electricFieldMatrix);
					return 1;
				}
				if (!gpuElectricFieldCalculator.UpdateDeviceChargesArrays())
				{
					CleanUp(electricFieldMatrix);
					return 1;
				}
				cpuElectricFieldCalculator.UpdateParameters(config.width, config.height, baseElectricForceMultiplier);

				imageCreator = ElectricFieldImageCreator(config.width, config.height, config.electricForceCoefficient, maxForce);
			}
		}

		chargesManager.UpdatePositions();

		if (config.isGpuModeEnabled && !gpuElectricFieldCalculator.SynchronizeDeviceAndCopyResult(electricFieldMatrix))
		{
			CleanUp(electricFieldMatrix);
			return 1;
		}

		imageCreator.UpdateImage(electricFieldMatrix);
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