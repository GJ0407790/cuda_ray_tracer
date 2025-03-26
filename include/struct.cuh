#ifndef RAY_H
#define RAY_H

#include <curand_kernel.h>
#include <math.h>

#include "vec3.cuh"
#include "interval.cuh"

/*r,g,b from 0 to 1*/
class RGB{
public:
	float r = 0.0f;
	float g = 0.0f;
	float b = 0.0f;

  __host__ __device__ RGB() {}
	__host__ __device__ RGB(float r, float g, float b): r(r), g(g), b(b){}
  
	__host__ __device__ bool operator==(const RGB& other) const 
	{
		return (r == other.r && g == other.g && b == other.b);
	}
  
	__host__ __device__ RGB operator-(const RGB& other) const 
	{
		return RGB(r - other.r, g - other.g, b - other.b);
	}
  
	__host__ __device__ RGB operator*(const RGB& other) const 
	{
		return RGB(r * other.r, g * other.g, b * other.b);
	}
};

/*r,g,b from 0 to 1*/
class RGBA{
public:
	float r = 0.0f;
	float g = 0.0f;
	float b = 0.0f;
	float a = 0.0f;

	__host__ __device__ RGBA() {}
	__host__ __device__ RGBA(float r,float g,float b,float a): r(r),g(g),b(b),a(a){}
	
	__host__ __device__ RGBA operator+(const RGBA& other) const 
	{
		return RGBA(r + other.r, g + other.g, b + other.b, a + other.a);
	}

  __host__ __device__ friend RGBA operator*(RGB rgb, const RGBA& other)
	{
		return RGBA(rgb.r * other.r, rgb.g * other.g, rgb.b * other.b, other.a);
	}

	__host__ __device__ RGBA mean(int aa)
	{
		float inv_aa = 1.0f / aa;
		return RGBA(r * inv_aa, g * inv_aa, b * inv_aa, a * inv_aa);
	}
};

#endif
