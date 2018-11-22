#pragma once
#include "iniFileStructure.h"


class INI_Reader
{
	bool isEmpty;

	SettingFile settings;

public:
	
	INI_Reader(const std::string& filePath);
	
	// returns value from section "groupName" assigned to "optionName"
	// if data is invalid, returns empty string
	std::string getValue(const std::string& groupName, const std::string& optionName) const;

	// returns false if Constructor could not open file
	bool isInitialized() const { return !isEmpty; }
};

