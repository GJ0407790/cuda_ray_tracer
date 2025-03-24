#ifndef RAY_H
#define RAY_H

#include <curand_kernel.h>
#include <math.h>

#include "vec3.cuh"
#include "interval.cuh"

/*r,g,b from 0 to 1*/
class RGB{
public:
	double r = 0.0;
	double g = 0.0;
	double b = 0.0;

  __host__ __device__ RGB(): r(0.0),g(0.0),b(0.0){}
	__host__ __device__ RGB(double r, double g, double b): r(r),g(g),b(b){}
  
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
	double r = 0.0;
	double g = 0.0;
	double b = 0.0;
	double a = 0.0;

	__host__ __device__ RGBA(): r(0.0),g(0.0),b(0.0),a(0.0){}
	__host__ __device__ RGBA(double r,double g,double b,double a): r(r),g(g),b(b),a(a){}
	
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
		double inv_aa = 1.0/aa;
		return RGBA(r * inv_aa, g * inv_aa, b * inv_aa, a * inv_aa);
	}
};

/*u,v*/
typedef struct{
	double u = 0;
	double v = 0;
}Texcoord;

#endif
