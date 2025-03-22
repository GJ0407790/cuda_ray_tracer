#ifndef BVH_H
#define BVH_H

#include <math.h>

#include "vec3.cuh"

class Interval {
public:
	double min,max;

	__host__ __device__ Interval() : min(+INFINITY), max(-INFINITY) {}

	__host__ __device__ Interval(double min,double max): min(min), max(max) {}

	__host__ __device__ double size() const {return max - min;}

	/**
	 * @brief Returns if the value is within, border value returns true.
	 * @param x The value to be checked.
	 * @return true 
	 * @return false 
	 */
	__host__ __device__ bool contains(double x) const {return min <= x && x <= max;}

	/**
	 * @brief Returns if the value is within, border value returns false.
	 * @param x The value to be checked.
	 * @return true 
	 * @return false 
	 */
	__host__ __device__ bool surrounds(double x) const {return min <= x && x <= max;}

	/**
	 * @brief Expand the interval by delta on both ends.
	 * @param delta The value to expand on both ends.
	 * @return Interval The expanded interval.
	 * @details This function is used for the bounding box creation for
	 *          triangles
	 */
	__host__ __device__ Interval expand(double delta) const {
		return Interval(min - delta, max + delta);
	}
};

#endif // BVH_H
