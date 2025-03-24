#include "config.hpp"
#include "helper.cuh"

#include <float.h> // got DBL_MAX
#include <math.h>

/**
 * @brief Translate color from linear RGB to sRGB.
 * @param l Linear RGB value. (0~1)
 * @return double sRGB value. (0~1)
 */
__host__ __device__ double RGBtosRGB(double l){
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
__host__ __device__ double sRGBtoRGB(double l){
	double c  = (l/255);
	return (c <= 0.04045f) ? (c / 12.92f) : pow((c + 0.055f) / 1.055f, 2.4f);
}

__device__ double setExpose(double c, RawConfig* config){
	if(config->expose == INT_MAX) return c;
	else return 1 - exp(-config->expose * c);
}

__device__ ObjectInfo unpackIntersection(const ObjectInfo& obj, const ObjectInfo& plane)
{
	//No object is intersecting this ray
	if(!obj.isHit && !plane.isHit) return ObjectInfo();

	double t1 = (obj.distance > 0) ? obj.distance : DBL_MAX;
	double t2 = (plane.distance > 0) ? plane.distance : DBL_MAX;
	double t = fmin(t1, t2);

	if(t == obj.distance)
	{
		return obj;
	}
	else if(t == plane.distance)
	{
		return plane;
	}
	else
	{
		printf("Error in function unpackIntersection");
		return ObjectInfo();
	}
}

__device__ BaryCenter getBarycentric(const Triangle& tri, const point3& p){
	// double b0,b1,b2;
	
	// b1 = dot(tri.e1, point - tri.p0);
	// b2 = dot(tri.e2, point - tri.p0);
	// b0 = 1.0 - b1 - b2;
	
	// return BaryCenter(b0, b1, b2); 

	vec3 v0 = tri.p1 - tri.p0;
	vec3 v1 = tri.p2 - tri.p0;
	vec3 v2 = p - tri.p0;

	double d00 = dot(v0, v0);
	double d01 = dot(v0, v1);
	double d11 = dot(v1, v1);
	double d20 = dot(v2, v0);
	double d21 = dot(v2, v1);

	double denom = d00 * d11 - d01 * d01;
	if (fabs(denom) < 1e-8) return BaryCenter(-1, -1, -1);  // degenerate triangle

	double v = (d11 * d20 - d01 * d21) / denom;
	double w = (d00 * d21 - d01 * d20) / denom;
	double u = 1.0 - v - w;

	return BaryCenter(u, v, w);
}

__device__ double randD(double start, double end, curandState* state) {
	double u = curand_uniform_double(state);  // (0.0, 1.0]
	return start + (end - start) * u;
}

__device__ double standerdD(double stddev, curandState* state) {
	return curand_normal_double(state) * stddev;
}

__device__ point3 spherePoint(curandState* state) 
{
    double z = 2.0 * randD(0, 1, state) - 1.0;
    double theta = 2.0 * 3.14159265 * randD(0, 1, state); 
    double r = sqrt(1.0 - z * z);

    double x = r * cos(theta);
    double y = r * sin(theta);

    return point3(x, y, z);
}
