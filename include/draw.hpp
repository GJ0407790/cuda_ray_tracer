/**
 * @file draw.hpp
 * @author Jin (jinj2@illinois.edu)
 * @brief 
 * @version 0.1
 * @date 2024-11-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef DRAW_H
#define DRAW_H

#include "libpng.h"
#include "helper.hpp"
#include "struct.hpp"

#include <vector>
#include <tuple>

void render(Image& img);

RGBA shootPrimaryRay(double x,double y);

ObjectInfo hitNearest(Ray& ray);

ObjectInfo hitMiss();

RGBA diffuseLight(const ObjectInfo& obj);

RGBA reflectionLight(const Ray& ray,const ObjectInfo& obj);

RGBA refractionLight(const Ray& ray,const ObjectInfo& obj);

RGBA globalIllumination(const ObjectInfo& obj,int gi_bounce);

ObjectInfo checkPlane(Ray& ray, bool exit_early = false);

RGBA getColorSun(double lambert,RGB objColor,RGB lightColor);

RGBA getColorBulb(double lambert,RGB objColor,RGB lightColor,double t);

RGBA getTexture();
#endif
