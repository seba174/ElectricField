#include "ConfigLoader.h"


ConfigLoader::ConfigLoader(const std::string& fileName)
	:configLoader(fileName)
{
}

int ConfigLoader::GetWidth() const
{
	std::string width = configLoader.getValue(GroupName, Width);
	if (width.empty())
		return DefaultWidth;
	try
	{
		return std::stoi(width);
	}
	catch (...)
	{
		return DefaultWidth;
	}
}

int ConfigLoader::GetHeight() const
{
	std::string height = configLoader.getValue(GroupName, Height);
	if (height.empty())
		return DefaultHeight;
	try
	{
		return std::stoi(height);
	}
	catch (...)
	{
		return DefaultHeight;
	}
}

int ConfigLoader::GetNumberOfCharges() const
{
	std::string numberOfCharges = configLoader.getValue(GroupName, NumberOfCharges);
	if (numberOfCharges.empty())
		return DefaultNumberOfCharges;
	try
	{
		return std::stoi(numberOfCharges);
	}
	catch (...)
	{
		return DefaultNumberOfCharges;
	}
}

float ConfigLoader::GetElectricForceCoefficient() const
{
	std::string electricForceCoefficient = configLoader.getValue(GroupName, ElectricForceCoefficient);
	if (electricForceCoefficient.empty())
		return DefaultElectricForceCoefficient;
	try
	{
		return std::stof(electricForceCoefficient);
	}
	catch (...)
	{
		return DefaultElectricForceCoefficient;
	}
}

int ConfigLoader::GetBlockSize() const
{
	std::string blockSize = configLoader.getValue(GroupName, BlockSize);
	if (blockSize.empty())
		return DefaultBlockSize;
	try
	{
		return std::stoi(blockSize);
	}
	catch (...)
	{
		return DefaultBlockSize;
	}
}

int ConfigLoader::GetMinVelocity() const
{
	std::string minVelocity = configLoader.getValue(GroupName, ChargeMinVelocity);
	if (minVelocity.empty())
		return DefaultMinVelocity;
	try
	{
		return std::stoi(minVelocity);
	}
	catch (...)
	{
		return DefaultMinVelocity;
	}
}

int ConfigLoader::GetMaxVelocity() const
{
	std::string maxVelocity = configLoader.getValue(GroupName, ChargeMaxVelocity);
	if (maxVelocity.empty())
		return DefaultMaxVelocity;
	try
	{
		return std::stoi(maxVelocity);
	}
	catch (...)
	{
		return DefaultMaxVelocity;
	}
}