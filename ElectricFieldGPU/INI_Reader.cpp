#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "INI_Reader.h"

using std::string;
using std::vector;

INI_Reader::INI_Reader(const string& filePath)
	: isEmpty(false)
{
	settings.filePath = filePath;

	std::ifstream in(filePath);
	if (!in)
	{
		std::cout << "Can't find config file!" << std::endl;
		isEmpty = true;
		return;
	}
	
	string line;
	int numberOfSettingGroups = 0;
	
	while (getline(in, line))
	{
		if (!line.empty())
		{
			size_t beg = line.find('[');
			size_t end = line.find(']');
			size_t tmp = line.find('%');
			if (beg != string::npos && end != string::npos)
			{
				// SettingGroup must have a form [*], where * is at least 1 character
				if (end > beg + 1)
				{
					++numberOfSettingGroups;
					settings.groups.push_back(SettingGroup());

					vector<SettingGroup>::iterator it = settings.groups.begin();
					std::advance(it, numberOfSettingGroups - 1);
					(*it).groupName = line.substr(beg + 1, end - (beg + 1));
				}
			}
			// if line has '%' inside, it means it is an comment!
			else if (tmp == string::npos && numberOfSettingGroups > 0)
			{
				vector<SettingGroup>::iterator it = settings.groups.begin();
				std::advance(it, numberOfSettingGroups - 1);
				size_t mid = line.find('=');

				// after '=' there must be at least 1 character
				if (mid != string::npos && mid != line.size() - 1)
				{
					SettingLine temp;
					temp.name = line.substr(0, mid);
					temp.value = line.substr(mid + 1);
					(*it).lines.push_back(temp);
				}
			}
		}
	}
	in.close();
}

string INI_Reader::getValue(const string & groupName, const string & optionName) const
{
	for (auto it = settings.groups.begin(); it != settings.groups.end(); ++it)
	{
		if (it->groupName == groupName)
		{
			for (auto tmp = it->lines.begin(); tmp != it->lines.end(); ++tmp)
			{
				if (tmp->name == optionName)
					return tmp->value;
			}
		}
	}
	// returns empty string if data is invalid
	return "";
}