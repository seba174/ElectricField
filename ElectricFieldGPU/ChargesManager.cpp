#include <stdlib.h>
#include "ChargesManager.h"


ChargesManager::ChargesManager(int numberOfCharges, int width, int height)
	:numberOfCharges(numberOfCharges), screenWidth(width), screenHeight(height)
{
	xCoordinate = new int[numberOfCharges];
	yCoordinate = new int[numberOfCharges];
	xVelocity = new int[numberOfCharges];
	yVelocity = new int[numberOfCharges];
}

void ChargesManager::SetRandomPositions()
{
	for (int i = 0; i < numberOfCharges; ++i)
	{
		xCoordinate[i] = rand() % screenWidth;
		yCoordinate[i] = rand() % screenHeight;
	}
}

void ChargesManager::SetRandomVelocities(int minVelocity, int maxVelocity)
{
	for (int i = 0; i < numberOfCharges; ++i)
	{
		xVelocity[i] = rand() % (maxVelocity - minVelocity) + minVelocity;
		yVelocity[i] = rand() % (maxVelocity - minVelocity) + minVelocity;
		if (rand() % 2 == 0)
			xVelocity[i] = -xVelocity[i];
		if (rand() % 2 == 0)
			yVelocity[i] = -yVelocity[i];
	}
}

void ChargesManager::UpdatePositions()
{
	for (int i = 0; i < numberOfCharges; ++i)
	{
		if (xCoordinate[i] + xVelocity[i] < 0)
		{
			xCoordinate[i] = 0;
			xVelocity[i] = -xVelocity[i];
		}
		else if (xCoordinate[i] + xVelocity[i] >= screenWidth)
		{
			xCoordinate[i] = screenWidth - 1;
			xVelocity[i] = -xVelocity[i];
		}
		else
		{
			xCoordinate[i] += xVelocity[i];
		}

		if (yCoordinate[i] + yVelocity[i] < 0)
		{
			yCoordinate[i] = 0;
			yVelocity[i] = -yVelocity[i];
		}
		else if (yCoordinate[i] + yVelocity[i] >= screenHeight)
		{
			yCoordinate[i] = screenHeight - 1;
			yVelocity[i] = -yVelocity[i];
		}
		else
		{
			yCoordinate[i] += yVelocity[i];
		}
	}
}

void ChargesManager::UpdateBounds(int newWidth, int newHeight)
{
	screenWidth = newWidth;
	screenHeight = newHeight;

	for (int i = 0; i < numberOfCharges; ++i)
	{
		if (xCoordinate[i] > screenWidth)
			xCoordinate[i] = screenWidth - 1;
		if (yCoordinate[i] > screenHeight)
			yCoordinate[i] = screenHeight - 1;
	}
}

int* ChargesManager::GetXCoordinates() const
{
	return xCoordinate;
}

size_t ChargesManager::GetXCoordinatesSize() const
{
	return numberOfCharges * sizeof(int);
}

int* ChargesManager::GetYCoordinates() const
{
	return yCoordinate;
}

size_t ChargesManager::GetYCoordinatesSize() const
{
	return numberOfCharges * sizeof(int);
}

int ChargesManager::GetNumberOfCharges() const
{
	return numberOfCharges;
}

ChargesManager::~ChargesManager()
{
	if (xCoordinate != nullptr)
		delete[] xCoordinate;
	if (yCoordinate != nullptr)
		delete[] yCoordinate;
	if (xVelocity != nullptr)
		delete[] xVelocity;
	if (yVelocity != nullptr)
		delete[] yVelocity;
}