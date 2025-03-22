#ifndef PARSE_H
#define PARSE_H

#include <string>
#include <sstream>
#include <vector>

#include "config.hpp"

void parseInput(char* argv[], StlConfig& config);

void parseLine(std::vector<std::string> words, StlConfig& config);

#endif // PARSE_H
