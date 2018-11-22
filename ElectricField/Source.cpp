#pragma once
#include <SFML/Graphics.hpp>
#include <iostream>
#include <math.h>
#include <fstream>
#include "INI_Reader.h"

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

	try
	{
		width = std::stoi(config.getValue(baseConfigGroup, "Width"));
		height = std::stoi(config.getValue(baseConfigGroup, "Height"));
		count = std::stoi(config.getValue(baseConfigGroup, "NumberOfCharges"));
		constK = std::stoi(config.getValue(baseConfigGroup, "ConstK"));
		toppercent = std::stof(config.getValue(baseConfigGroup, "TopPercent"));
		botpercent = std::stof(config.getValue(baseConfigGroup, "BotPercent"));
	}
	catch (std::invalid_argument exception) {}


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

	sf::RenderWindow window;
	window.create(sf::VideoMode(width, height), "", sf::Style::Default);

	sf::Image image;
	sf::Texture texture;
	sf::Sprite sprite;
	image.create(width, height);

	sf::Clock clock;
	int fps = 0;

	//float maxForce = 100 * (constK / 0.01);

	while (window.isOpen())
	{
		//std::cout << sf::Mouse::getPosition(window).x << " " << sf::Mouse::getPosition(window).y << std::endl;
		if (clock.getElapsedTime().asSeconds() >= 25)
		{
			std::cout << fps << std::endl;
			fps = 0;
			clock.restart();
		}
		/*for (int i = 0; i < count; i++)
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
		}*/

		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}

		for (int dim = 0; dim < pixelArraySize; ++dim)
		{
			int w = dim % width;
			int h = dim / width;
			float forceX = 0, forceY = 0;
			for (int k = 0; k < count; ++k)
			{
				int dx = w - x[k];
				int dy = h - y[k];
				float magnitude = sqrtf(dx * dx + dy * dy) + 0.01;
				float chargefm = constK / magnitude;

				float xUnit = dx / magnitude;
				float yUnit = dy / magnitude;

				forceX += abs(xUnit * chargefm);
				forceY += abs(yUnit * chargefm);
			}

			forceInPixels[dim] = sqrtf(forceX * forceX + forceY * forceY);
		}

		float maxForce = 0;
		for (int i = 0; i < pixelArraySize; ++i)
		{
			if (forceInPixels[i] > maxForce)
				maxForce = forceInPixels[i];
		}

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

	delete[](forceInPixels);
	delete[](x);
	delete[](y);

	return 0;
}


