/**
 * @file helper.hpp
 * @author Jin (jinj2@illinois.edu)
 * @brief 
 * @version 0.1
 * @date 2024-11-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef HELPER_H
#define HELPER_H
#include <string>
#include <tuple>
#include <cmath>
#include "vec3.hpp"
#include "struct.hpp"
#include "object.hpp"


void print(std::string str);

/*The color functions*/
double RGBtosRGB(double l);
double sRGBtoRGB(double l);
void setImageColor(Image& img,RGBA rgba,int x,int y);
double setExpose(double c);

ObjectInfo unpackIntersection(const ObjectInfo& sphere,
							const ObjectInfo& plane);
std::tuple<double,double,double> getBarycentric(const Triangle& tri,const point3& point);

double randD(double start,double end);

double standerdD(double stddev);

point3 spherePoint();

void print(int i);
void print(double f);

void printErr(std::string str);

#endif
