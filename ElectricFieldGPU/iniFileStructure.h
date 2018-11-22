#pragma once
#include <string>
#include <vector>

class SettingLine
{
public:
	SettingLine() {}
	SettingLine(const std::string& name, const std::string& value) :name(name), value(value) {}
	std::string name;
	std::string value;
};

class SettingGroup
{
public:
	std::string groupName;
	std::vector<SettingLine> lines;
};

class SettingFile
{
public:
	std::string filePath;
	std::vector<SettingGroup> groups;
};

