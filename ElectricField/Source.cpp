#pragma once
#include <SFML/Graphics.hpp>
#include <iostream>
#include <math.h>

int main()
{
	srand(100);

	int width = 1200;
	int height = 800;
	int count = 10;

	const int constK = 100;
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

	sf::Clock clock;
	int fps = 0;

	while (window.isOpen())
	{
		if (clock.getElapsedTime().asSeconds() >= 1)
		{
			std::cout << fps << std::endl;
			fps = 0;
			clock.restart();
		}

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

				forceX += xUnit * chargefm;
				forceY += yUnit * chargefm;
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
		float toppercent = 0.5;
		float botpercent = 0;
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


