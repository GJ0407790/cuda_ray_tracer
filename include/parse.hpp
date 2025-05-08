#ifndef PARSE_H
#define PARSE_H

#include <string>
#include <sstream>
#include <vector>

#include "config.hpp"

void parseInput(char* argv[], StlConfig& config);

void parseLine(std::vector<std::string> words, StlConfig& config, AABB& running_scene_bounds);

#endif // PARSE_H
