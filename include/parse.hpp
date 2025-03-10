/**
 * @file parse.hpp
 * @author Jin (jinj2@illinois.edu)
 * @brief 
 * @version 0.1
 * @date 2024-11-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef PARSE_H
#define PARSE_H

#include <string>
#include <sstream>
#include <vector>

#include "config.hpp"

void parseInput(char* argv[], Config& config);

void parseLine(std::vector<std::string> words, Config& config);

#endif
