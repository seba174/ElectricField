#include "ElectricFieldImageCreator.h"


int ElectricFieldImageCreator::differentColorsCount = 1149;

ElectricFieldImageCreator::ElectricFieldImageCreator(int screenWidth, int screenHeight, float electricForceCoefficient, float maxElectricForce)
	:screenWidth(screenWidth), screenHeight(screenHeight), electricForceCoefficient(electricForceCoefficient), maxElectricForce(maxElectricForce)
{
	image.create(screenWidth, screenHeight);
}

void ElectricFieldImageCreator::UpdateImage(float * electricFieldMatrix)
{
	float top = (1 - electricForceCoefficient) * maxElectricForce;
	float divider = top / differentColorsCount;

	for (int i = 0; i < screenWidth; ++i)
	{
		for (int j = 0; j < screenHeight; ++j)
		{
			float level = (electricFieldMatrix[i + j * screenWidth]) / divider;

			if (level <= 0)
				image.setPixel(i, j, sf::Color(0, 0, 0));
			else if (level <= 255)
				image.setPixel(i, j, sf::Color(0, 0, static_cast<int>(level)));
			else if (level <= 510)
				image.setPixel(i, j, sf::Color(0, static_cast<int>(level - 255), 255));
			else if (level <= 765)
				image.setPixel(i, j, sf::Color(static_cast<int>(level - 510), 255, static_cast<int>(765 - level)));
			else if (level <= 1020)
				image.setPixel(i, j, sf::Color(255, static_cast<int>(1020 - level), 0));
			else if (level <= 1149)
				image.setPixel(i, j, sf::Color(static_cast<int>(1276 - level), 0, 0));
			else
				image.setPixel(i, j, sf::Color(127, 0, 0));
		}
	}
}

const sf::Sprite & ElectricFieldImageCreator::GetSprite()
{
	texture.loadFromImage(image);
	sprite.setTexture(texture);

	return sprite;
}