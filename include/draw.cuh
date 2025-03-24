#ifndef DRAW_H
#define DRAW_H

#include "helper.cuh"
#include "struct.cuh"
#include "config.hpp"

#include <curand_kernel.h> // for device rng

void render(pixel_t* d_image, int img_width, int img_height, int aa, RawConfig* config);

__device__ RGBA shootPrimaryRay(double x, double y, curandState* state, RawConfig* config);

__device__ ObjectInfo hitNearest(Ray& ray, RawConfig* config);

__device__ ObjectInfo hitMiss();

__device__ RGBA diffuseLight(const ObjectInfo& obj, curandState* state, RawConfig* config);

__device__ RGBA reflectionLight(const Ray& ray,const ObjectInfo& obj, curandState* state, RawConfig* config);

__device__ RGBA refractionLight(const Ray& ray,const ObjectInfo& obj, curandState* state, RawConfig* config);

__device__ RGBA globalIllumination(const ObjectInfo& obj,int gi_bounce, curandState* state, RawConfig* config);

__device__ ObjectInfo checkPlane(Ray& ray, bool exit_early, RawConfig* config);

__device__ RGBA getColorSun(double lambert,RGB objColor,RGB lightColor, RawConfig* config);

__device__ RGBA getColorBulb(double lambert, RGB objColor, RGB lightColor, double t, RawConfig* config);

#endif
