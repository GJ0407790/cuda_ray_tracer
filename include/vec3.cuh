#ifndef VEC3_H
#define VEC3_H

#include <math.h>
#include <stdio.h>

/*vector direction of a ray*/
class vec3{
public:
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;

	__host__ __device__ vec3(){};

	__host__ __device__ vec3(double a,double b,double c): x(a),y(b),z(c) {};

	__host__ __device__ bool operator==(const vec3& other) const {
		return (x == other.x && y == other.y && z==other.z);
	}

	__host__ __device__ double operator[](size_t index) const{
		if (index == 0) return x;
		else if (index == 1) return y;
		else if (index == 2) return z;

		printf("Out of bounds for %zu\n", index);
		return 0.0;
  }

	__host__ __device__ vec3 operator+(const vec3& other) const {
		return vec3(x+other.x,y+other.y,z+other.z);
	}

	__host__ __device__ vec3 operator-(const vec3& other) const {
		return vec3(x-other.x,y-other.y,z-other.z);
	}

	__host__ __device__ vec3 operator*(double scalar) const {
		return vec3(x * scalar,y * scalar,z * scalar);
	}

	__host__ __device__ vec3 operator/(double scalar) const {
		return vec3(x / scalar,y / scalar,z / scalar);
	}

	__host__ __device__ friend vec3 operator*(double scalar, const vec3& other) {
		return vec3(other.x * scalar, other.y * scalar, other.z * scalar);
	}

	__host__ __device__ double dot(const vec3& other) const {
		return x * other.x + y * other.y + z * other.z;
	}

	__host__ __device__ double length() const {
		return sqrt(x * x + y * y + z * z);
	}

	__host__ __device__ vec3 normalize() const {
		double mag = length();
		if (mag == 0) {
			return vec3(0.0, 0.0, 0.0);
		}

		double inv_mag = 1 / mag;
		return vec3(x * inv_mag, y * inv_mag, z * inv_mag);
	}

	/**
	 * @brief Inverts the vector
	 * @return vec3 
	 */
	__host__ __device__ vec3 operator-() const {
		return vec3(-x, -y, -z);
	}
};

using point3 = vec3;

__host__ __device__ inline double dot(const vec3& a, const vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline vec3 cross(const vec3& a, const vec3& b) {
	return vec3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

#endif
