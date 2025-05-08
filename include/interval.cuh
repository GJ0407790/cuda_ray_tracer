#ifndef BVH_H
#define BVH_H

#include <math.h>

#include "vec3.cuh"

class Interval {
public:
	float min,max;

	__host__ __device__ Interval() : min(+INFINITY), max(-INFINITY) {}

	__host__ __device__ Interval(float min,float max): min(min), max(max) {}

	__host__ __device__ float size() const {return max - min;}

	/**
	 * @brief Returns if the value is within, border value returns true.
	 * @param x The value to be checked.
	 * @return true 
	 * @return false 
	 */
	__host__ __device__ bool contains(float x) const {return min <= x && x <= max;}

	/**
	 * @brief Returns if the value is within, border value returns false.
	 * @param x The value to be checked.
	 * @return true 
	 * @return false 
	 */
	__host__ __device__ bool surrounds(float x) const {return min <= x && x <= max;}

	/**
	 * @brief Expand the interval by delta on both ends.
	 * @param delta The value to expand on both ends.
	 * @return Interval The expanded interval.
	 * @details This function is used for the bounding box creation for
	 *          triangles
	 */
	__host__ __device__ Interval expand(float delta) const {
		return Interval(min - delta, max + delta);
	}
};

class AABB {
public:
	Interval x,y,z;

	__host__ __device__ AABB() {}

	__host__ __device__ AABB(const Interval& x,const Interval& y,const Interval& z):
			x(x),y(y),z(z) {}
	
	__host__ __device__ AABB(const point3& a, const point3& b) {
		x = (a.x <= b.x) ? Interval(a.x, b.x) : Interval(b.x, a.x);
		y = (a.y <= b.y) ? Interval(a.y, b.y) : Interval(b.y, a.y);
		z = (a.z <= b.z) ? Interval(a.z, b.z) : Interval(b.z, a.z);
	}

	__host__ __device__ AABB(const point3& a,const point3& b,const point3& c) {		
		//obtain the min and max value for the three coordinates
		float x_min = fmin(fmin(a.x, b.x), c.x);
		float x_max = fmax(fmax(a.x, b.x), c.x);

		float y_min = fmin(fmin(a.y, b.y), c.y);
		float y_max = fmax(fmax(a.y, b.y), c.y);

		float z_min = fmin(fmin(a.z, b.z), c.z);
		float z_max = fmax(fmax(a.z, b.z), c.z);

		// Create intervals
		x = Interval(x_min, x_max);
		y = Interval(y_min, y_max);
		z = Interval(z_min, z_max);
		
		//expand if the interval is too small
		if(x.size() < 0.01f) x = x.expand(0.01f);
		if(y.size() < 0.01f) y = y.expand(0.01f);
		if(z.size() < 0.01f) z = z.expand(0.01f);
	}

	__host__ __device__ AABB(const AABB& a, const AABB& b) 
	{
		x = Interval(fmin(a.x.min, b.x.min), fmax(a.x.max, b.x.max));
		y = Interval(fmin(a.y.min, b.y.min), fmax(a.y.max, b.y.max));
		z = Interval(fmin(a.z.min, b.z.min), fmax(a.z.max, b.z.max));
	}

	__host__ __device__ const Interval& getAxis(int n) const 
	{
		if (n == 0) return x;
		if (n == 1) return y;
		return z;
	}

	__host__ __device__ int longestAxis() const 
	{
		if (x.size() > y.size())
			return x.size() > z.size() ? 0 : 2;
		else
			return y.size() > z.size() ? 1 : 2;
	}
};

#endif // BVH_H
