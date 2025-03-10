/**
 * @file main.cpp
 * @author Jin (jinj2@illinois.edu)
 * @brief 
 * @version 0.1
 * @date 2024-11-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <iostream>

#include "include/config.hpp"
#include "include/draw.hpp"
#include "include/parse.hpp"

using std::cout;
using std::endl;

Config config;

int main(int argc, char* argv[]){
    if(argc != 2){
        std::cerr << "Use case: make run file=your/file.txt" << endl;
        exit(1);
    }

    //parse the inputs
    parseInput(argv, config);
    
    //create the image
    Image img(config.width, config.height);
    
    render(img);
    
    img.save(config.filename.c_str());
}
