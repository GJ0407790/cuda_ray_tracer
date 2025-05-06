#include "config.hpp"
#include "helper.cuh"

#include <float.h> // got DBL_MAX
#include <math.h>

/**
 * @brief Translate color from linear RGB to sRGB.
 * @param l Linear RGB value. (0~1)
 * @return float sRGB value. (0~1)
 */
__host__ __device__ float RGBtosRGB(float l){
	float sol;

	if(l < 0.0031308f)
	{
		sol = 12.92f * l;
	}
	else
	{	
		sol = 1.055f * pow(l, 1/2.4f) - 0.055f;
	}

	sol = fmin(1.0f, fmax(0.0f, sol));

	return sol;
}

/**
 * @brief Translate color from sRGB to linear RGB.
 * @param l sRGB value. (0~255)
 * @return float Linear RGB value.(0~1)
 */
__host__ __device__ float sRGBtoRGB(float l)
{
	float c  = (l/255.0f);
	return (c <= 0.04045f) ? (c / 12.92f) : pow((c + 0.055f) / 1.055f, 2.4f);
}

__device__ float setExpose(float c, RawConfig* config)
{
	return config->expose == float(INFINITY) 
						? c 
						: 1.0 - exp(-config->expose * c);
}

__device__ ObjectInfo unpackIntersection(const ObjectInfo& obj, const ObjectInfo& plane)
{
	const bool valid_obj = obj.isHit && obj.distance > 0.0f && !isnan(obj.distance);
	const bool valid_plane = plane.isHit && plane.distance > 0.0f && !isnan(plane.distance);

	if (!valid_obj && !valid_plane) 
	{
		return ObjectInfo();  // no valid intersections
	}

	if (valid_obj && (!valid_plane || obj.distance < plane.distance)) 
	{
		return obj;
	}

	if (valid_plane) 
	{
		return plane;
	}

	// fallback, should never reach here
	printf("unpackIntersection error: obj=%.5f, plane=%.5f\n", obj.distance, plane.distance);
	return ObjectInfo();
}

__device__ BaryCenter getBarycentric(const Triangle& tri, const point3& point){
	float b0,b1,b2;
	
	b1 = dot(tri.e1, point - tri.p0);
	b2 = dot(tri.e2, point - tri.p0);
	b0 = 1.0f - b1 - b2;
	
	return BaryCenter(b0, b1, b2); 
}

__device__ float randD(float start, float end, curandState* state) {
	float u = curand_uniform(state);  // (0.0, 1.0]
	return start + (end - start) * u;
}

__device__ float standerdD(float stddev, curandState* state) {
	return curand_normal(state) * stddev;
}

__device__ point3 spherePoint(curandState* state) 
{
    float z = 2.0f * randD(0.0f, 1.0f, state) - 1.0f;
    float theta = 2.0f * 3.14159265f * randD(0.0f, 1.0f, state); 
    float r = sqrt(1.0f - z * z);

    float x = r * cos(theta);
    float y = r * sin(theta);

    return point3(x, y, z);
}
