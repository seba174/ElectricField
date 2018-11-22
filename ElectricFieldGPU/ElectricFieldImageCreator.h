#pragma once
#include <SFML/Graphics.hpp>


class ElectricFieldImageCreator
{
private:
	static int differentColorsCount;

	sf::Image image;
	sf::Texture texture;
	sf::Sprite sprite;

	float electricForceCoefficient;
	int screenWidth;
	int screenHeight;
	float maxElectricForce;


public:

	ElectricFieldImageCreator(int screenWidth, int screenHeight, float electricForceCoefficient, float maxElectricForce);
	
	void UpdateImage(float* electricFieldMatrix);

	const sf::Sprite& GetSprite();
};