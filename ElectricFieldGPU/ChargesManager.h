#pragma once

class ChargesManager
{
private:
	int numberOfCharges;
	int* xCoordinate;
	int* yCoordinate;
	int* xVelocity;
	int* yVelocity;

public:
	int screenWidth;
	int screenHeight;
	int minVelocity;
	int maxVelocity;

	ChargesManager(int numberOfCharges);

	void SetRandomPositions();

	void SetRandomVelocities();

	void UpdatePositions();

	int* GetXCoordinates() const;

	size_t GetXCoordinatesSize() const;

	int* GetYCoordinates() const;

	size_t GetYCoordinatesSize() const;

	int GetNumberOfCharges();

	~ChargesManager();
};