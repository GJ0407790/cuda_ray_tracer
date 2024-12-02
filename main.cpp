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
#include "include/all.hpp"
#include <iostream>
using std::cout;
using std::endl;

int main(int argc, char* argv[]){
    if(argc != 2){
        std::cerr << "Use case: make run file=your/file.txt" << endl;
        exit(1);
    }
    //parse the inputs
    parseInput(argv);
    
    //create the image
    Image img(width,height);
    
    render(img);
    
    img.save(filename.c_str());
}
