#pragma once
#include "INI_Reader.h"

class ConfigLoader
{
private:
	const std::string GroupName = "Config";
	const std::string Width = "Width";
	const std::string Height = "Height";
	const std::string NumberOfCharges = "NumberOfCharges";
	const std::string ElectricForceCoefficient = "ElectricForceCoefficient";
	const std::string BlockSize = "BlockSize";
	const std::string ChargeMinVelocity = "ChargeMinVelocity";
	const std::string ChargeMaxVelocity = "ChargeMaxVelocity";
	const std::string GpuModeEnabled = "GpuModeEnabled";
	const int DefaultWidth = 800;
	const int DefaultHeight = 600;
	const int DefaultNumberOfCharges = 50;
	const float DefaultElectricForceCoefficient = 0.6f;
	const int DefaultBlockSize = 512;
	const int DefaultMinVelocity = 1;
	const int DefaultMaxVelocity = 5;
	const bool DefaultGpuModeEnabled = true;

	std::string fileName;
	INI_Reader configLoader;

public:
	int minSpeed;
	int maxSpeed;
	int width;
	int height;
	int numberOfCharges;
	float electricForceCoefficient;
	int blockSize;
	bool isGpuModeEnabled;

	ConfigLoader(const std::string& fileName);

	void LoadConfig();

	int GetWidth() const;
	int GetHeight() const;
	int GetNumberOfCharges() const;
	float GetElectricForceCoefficient() const;
	int GetBlockSize() const;
	int GetMinVelocity() const;
	int GetMaxVelocity() const;
	bool GetGpuModeEnabled() const;
};