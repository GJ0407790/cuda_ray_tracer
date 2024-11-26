/**
 * @file vec3.hpp
 * @author Jin (jinj2@illinois.edu)
 * @brief 
 * @version 0.1
 * @date 2024-11-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef VEC3_H
#define VEC3_H
#include <cmath>
#include <iostream>

/*vector direction of a ray*/
class vec3{
public:
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;

	vec3(){};

	vec3(double a,double b,double c): x(a),y(b),z(c) {};

	bool operator==(const vec3& other) const {
		return (x == other.x && y == other.y && z==other.z);
	}

	double operator[](size_t index) const{
        if (index == 0) return x;
        else if (index == 1) return y;
        else if (index == 2) return z;
        throw std::out_of_range("Index out of range");
    }


	vec3 operator+(const vec3& other) const {
		return vec3(x+other.x,y+other.y,z+other.z);
	}

	vec3 operator-(const vec3& other) const {
		return vec3(x-other.x,y-other.y,z-other.z);
	}

	vec3 operator*(double scalar) const {
		return vec3(x * scalar,y * scalar,z * scalar);
	}

	vec3 operator/(double scalar) const {
		return vec3(x / scalar,y / scalar,z / scalar);
	}

	friend vec3 operator*(double scalar, const vec3& other) {
		return vec3(other.x * scalar, other.y * scalar, other.z * scalar);
	}

	double dot(const vec3& other) const {
		return x * other.x + y * other.y + z * other.z;
	}

	double length() const {
		return std::sqrt(x * x + y * y + z * z);
	}

	vec3 normalize() const {
	double mag = length();
	if (mag == 0) {
		return vec3(0.0, 0.0, 0.0);
	}
	return vec3(x / mag, y / mag, z / mag);
	}

	/**
	 * @brief Inverts the vector
	 * @return vec3 
	 */
	vec3 operator-() const {
		return vec3(-x, -y, -z);
	}
};

using point3 = vec3;

inline double dot(const vec3& a, const vec3& b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline vec3 cross(const vec3& a, const vec3& b) {
	return vec3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}

#endif
