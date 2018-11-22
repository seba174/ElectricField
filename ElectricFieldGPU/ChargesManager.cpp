#include <stdlib.h>
#include "ChargesManager.h"


ChargesManager::ChargesManager(int numberOfCharges)
	:numberOfCharges(numberOfCharges)
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

void ChargesManager::SetRandomVelocities()
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

int* ChargesManager::GetXCoordinates() const
{
	return xCoordinate;
}

size_t ChargesManager::GetXCoordinatesSize() const
{
	return numberOfCharges * sizeof(*xCoordinate);
}

int* ChargesManager::GetYCoordinates() const
{
	return yCoordinate;
}

size_t ChargesManager::GetYCoordinatesSize() const
{
	return numberOfCharges * sizeof(*yCoordinate);
}

int ChargesManager::GetNumberOfCharges()
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
