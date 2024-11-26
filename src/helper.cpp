/**
 * @file helper.cpp
 * @author Jin (jinj2@illinois.edu)
 * @brief 
 * @version 0.1
 * @date 2024-11-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "../include/all.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <climits>
#include <random>

using namespace std;

/** 
 * @brief print a string
 * 
*/
void print(string str){
	cout << str << endl;
}


void print(int i){
	cout << to_string(i) << endl;
}

void print(double f){
	cout << to_string(f) << endl;
}

void printErr(string str){
	cout << "\033[31m" << str << "\033[0m" << endl;
}

/**
 * @brief Translate color from linear RGB to sRGB.
 * @param l Linear RGB value. (0~1)
 * @return double sRGB value. (0~1)
 */
double RGBtosRGB(double l){
	double sol;
	if(l < 0.0031308)
		sol = 12.92 * l;
	else
		sol = 1.055 * pow(l,1/2.4) -0.055;
	if(sol < 0) sol = 0.0;
	if(sol > 1) sol = 1.0f;
	return sol;
}

/**
 * @brief Translate color from sRGB to linear RGB.
 * @param l sRGB value. (0~255)
 * @return double Linear RGB value.(0~1)
 */
double sRGBtoRGB(double l){
	double c  = (l/255);
	return (c <= 0.04045f) ? (c / 12.92f) : pow((c + 0.055f) / 1.055f, 2.4f);
}

/**
 * @brief Set the Image Color at pixel (x,y) based on the RGBA input.
 * @param img The image to set.
 * @param rgba The color to use, in linear RGB color space.
 * @param x x coordinate of the pixel.
 * @param y y coordinate of the pixel.
 */
void setImageColor(Image& img,RGBA rgba,int x,int y){
	auto[r,g,b,a] = rgba;
	img[y][x].r = RGBtosRGB(r) * 255;
	img[y][x].g = RGBtosRGB(g) * 255;
	img[y][x].b = RGBtosRGB(b) * 255;
	img[y][x].a = a * 255;
}

double setExpose(double c){
	if(expose == INT_MAX) return c;
	else return 1 - std::exp(-expose * c);
}

ObjectInfo unpackIntersection(const ObjectInfo& obj,
							const ObjectInfo& plane){
	double t;
	//No object is intersecting this ray
	if(!obj.isHit && !plane.isHit) return ObjectInfo();

	t = std::min({obj.distance > 0 ? obj.distance : std::numeric_limits<double>::max(),
				plane.distance > 0 ? plane.distance : std::numeric_limits<double>::max()});

	if(t == obj.distance){
		return obj;
	}else if(t == plane.distance){
		return plane;
	}else{
		print("Error in function unpackIntersection");
		return ObjectInfo();
	}
}

std::tuple<double,double,double> getBarycentric(const Triangle& tri,const point3& point){
	double b0,b1,b2;
	b1 = dot(tri.e1,point - tri.p0);
	b2 = dot(tri.e2,point - tri.p0);
	b0 = 1 - b1 - b2;
	return std::make_tuple(b0,b1,b2); 
}

double randD(double start,double end){
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(start,end);
	return dis(gen);
}

double standerdD(double stddev){
	std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0,stddev);
	return d(gen);
}

point3 spherePoint() {
    double z = 2.0 * randD(0,1) - 1.0;
    double theta = 2.0 * 3.14159265 * randD(0,1); 
    double r = sqrt(1.0 - z * z);

    double x = r * cos(theta);
    double y = r * sin(theta);
    return point3(x, y, z);
}
