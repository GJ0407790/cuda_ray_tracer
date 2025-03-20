#ifndef DRAW_H
#define DRAW_H

#include "libpng.h"
#include "helper.hpp"
#include "struct.hpp"

#include <vector>
#include <tuple>

void render(RGBA* d_image, int img_width, int img_height, int aa);

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
