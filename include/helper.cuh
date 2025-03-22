#ifndef HELPER_HPP
#define HELPER_HPP

#include "vec3.cuh"
#include "struct.cuh"
#include "object.cuh"
#include "config.hpp"

/*The color functions*/
__host__ __device__ double RGBtosRGB(double l);
__host__ __device__ double sRGBtoRGB(double l);
__device__ double setExpose(double c, RawConfig* config);
__device__ ObjectInfo unpackIntersection(const ObjectInfo& sphere, const ObjectInfo& plane);

class BaryCenter
{
public:
  double b0;
  double b1;
  double b2;

  __host__ __device__ BaryCenter(double b0, double b1, double b2) : b0{b0}, b1{b1}, b2{b2} {}
};

__device__ BaryCenter getBarycentric(const Triangle& tri,const point3& point);

__device__ double randD(double start, double end, curandState* state);
__device__ double standerdD(double stddev, curandState* state);

__device__ point3 spherePoint(curandState* state);

#endif
