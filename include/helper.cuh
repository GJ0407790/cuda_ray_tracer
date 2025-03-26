#ifndef HELPER_HPP
#define HELPER_HPP

#include "vec3.cuh"
#include "struct.cuh"
#include "object.cuh"
#include "config.hpp"

/*The color functions*/
__host__ __device__ float RGBtosRGB(float l);
__host__ __device__ float sRGBtoRGB(float l);
__device__ float setExpose(float c, RawConfig* config);
__device__ ObjectInfo unpackIntersection(const ObjectInfo& sphere, const ObjectInfo& plane);

class BaryCenter
{
public:
  float b0;
  float b1;
  float b2;

  __host__ __device__ BaryCenter(float b0, float b1, float b2) : b0{b0}, b1{b1}, b2{b2} {}
};

__device__ BaryCenter getBarycentric(const Triangle& tri, const point3& point);

__device__ float randD(float start, float end, curandState* state);
__device__ float standerdD(float stddev, curandState* state);

__device__ point3 spherePoint(curandState* state);

#endif
