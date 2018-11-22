#pragma once

class ChargesManager
{
private:
	int numberOfCharges;
	int screenWidth;
	int screenHeight;

	int* xCoordinate;
	int* yCoordinate;
	int* xVelocity;
	int* yVelocity;

	void CleanUp();
	void Init();

public:
	ChargesManager(int numberOfCharges, int width, int height);

	void UpdateSize(int numberOfCharges, int width, int height);

	void SetRandomPositions();

	void SetRandomVelocities(int minVelocity, int maxVelocity);

	void UpdatePositions();

	void UpdateBounds(int newWidth, int newHeight);

	int* GetXCoordinates() const;

	size_t GetXCoordinatesSize() const;

	int* GetYCoordinates() const;

	size_t GetYCoordinatesSize() const;

	int GetNumberOfCharges() const;

	~ChargesManager();
};