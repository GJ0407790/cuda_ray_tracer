#ifndef OBJECT_H
#define OBJECT_H

#include "struct.cuh"
#include "config.hpp"
#include "libpng.h"

#include <algorithm>
#include <memory>

class RawConfig;

/**
 * @brief Material related variables for objects in the scene.
 * 		  
 */
class Materials{
public:
	RGB color; //!< The color, in RGB format, of the material
	RGB shininess; //!< The shininess of the object, which is related to reflection. 
	RGB trans; 	//! < The transparency of the object, which is related to refraction. 
	float ior = 1.458f; //!< Index of refraction. Default to 1.458.
	float roughness = 0.0f; //!< Roughness of the object, Default to zero(none).
	/**
	 * @brief Construct a new Materials object, with every member as default.
	 */
	__host__ __device__ Materials() {}
	/**
	 * @brief Construct a new Materials object, with inputs.
	 * @param color 
	 * @param shininess 
	 * @param trans 
	 * @param ior 
	 * @param roughness 
	 */
	__host__ __device__ Materials(RGB color, RGB shininess, RGB trans, float ior, float roughness):
		color(color),shininess(shininess),trans(trans),ior(ior),roughness(roughness) {}
};

/**
 * @brief ObjectInfo is a class that passes the parameters of a ray hit, and the
 * 		details of the object that was hit.
 */
class ObjectInfo{
public:
	bool isHit = false; //!< Signals if there was a hit. Defaults to false.
	float distance; //!< The distance from the ray origin to the object.
	point3 i_point;  //!< The intersection point.
	vec3 normal; //!< The normal at the point of intersection.
	Materials mat; //!< Material properties.
	
	__device__ ObjectInfo(): distance(-1.0f),i_point(point3()),normal(vec3()),mat(Materials()) {}
  __device__ ObjectInfo(float distance, point3 i_point, vec3 normal,Materials mat)
							: isHit(true), distance(distance), i_point(i_point), 
							  normal(normal), mat(mat) {}
};

/**
 * @brief Class Ray, consists of a eye and direction
 */
class Ray{
public:
	point3 eye;
	vec3 dir;
	int bounce;

	__device__ Ray() : bounce(0) {}
	__device__ Ray(float x, float y, curandState* state, RawConfig* config);
	__device__ Ray(point3 eye, vec3 dir, int bounce): eye(eye), dir(dir.normalize()),bounce(bounce){}
};

// __device__ bool is_intersect(const Ray& r, const AABB& bbox) 
// {
// 	const point3& ray_eye = r.eye;
// 	const vec3&   ray_dir = r.dir;

// 	float t_min = float(-INFINITY);
// 	float t_max = float(INFINITY);

// 	for (int axis = 0; axis < 3; axis++) 
// 	{
// 		const Interval& ax = bbox.getAxis(axis);
		
// 		if (ax.min > ax.max) 
// 		{
// 			return false;
// 		}

// 		const float adinv = 1.0f / ray_dir[axis];

// 		float t0 = (ax.min - ray_eye[axis]) * adinv;
// 		float t1 = (ax.max - ray_eye[axis]) * adinv;

// 		// Manual swap
// 		if (t0 > t1) {
// 			float temp = t0;
// 			t0 = t1;
// 			t1 = temp;
// 		}

// 		if (t0 > t_min)
// 		{
// 			t_min = t0;
// 		}

// 		if (t1 < t_max)
// 		{
// 			t_max = t1;
// 		}

// 		if (t_max <= t_min) 
// 		{
// 			return false;
// 		}
// 	}

// 	return true;
// }

enum class PrimitiveType 
{
	SPHERE,
	TRIANGLE
	// Add other types if you have them, e.g., MESH_INSTANCE
};

struct PrimitiveReference 
{
	PrimitiveType type;
	unsigned int id_in_type_array; // Index into d_all_spheres or d_all_triangles
	// Optional: you might add original scene object ID for debugging, or material ID
	// unsigned int material_idx;

	__host__ __device__ PrimitiveReference() : type(PrimitiveType::SPHERE), id_in_type_array(0) {} // Default constructor
	__host__ __device__ PrimitiveReference(PrimitiveType t, unsigned int id) : type(t), id_in_type_array(id) {}
};

/**
 * @brief Class sphere, with one center point 
 *        and a radius, together with a rgb color
 *        value.
 */
class Sphere {
public:
	struct UV 
	{
		float u;
		float v;

		__device__ UV(float u_val, float v_val)
								: u(u_val), v(v_val) {}
	};

public:
	point3 c; //!< Center point of the sphere.
	float r; //!< Radius of the sphere.
	Materials mat; //!< Material properties.
	AABB bbox; //!< Axis-aligned bounding box for the Object class. For BVH traversal.

	/**
	 * @brief Construct a new Sphere object with no inputs
	 */
	Sphere(): r(0.0f) {
		c = point3(0.0f, 0.0f, 0.0f);
		mat.color = {1.0f, 1.0f, 1.0f};
	}

	/**
	 * @brief Construct a new Sphere object, with color inputs.
	 */
	Sphere(float x, float y, float z, float r, RGB rgb): r(r) 
	{
		c = point3(x, y, z);
		mat.color = rgb;
		auto rvec = vec3(r, r, r);
    bbox = AABB(c - rvec, c + rvec);
	}

	__host__ __device__ void setProperties(RGB shine, RGB tran, float ior, float roughness) 
	{
		mat.shininess = shine;
		mat.trans = tran;
		mat.ior = ior;
		mat.roughness = roughness;
	}

	__host__ __device__ AABB getBox() const 
	{
		return bbox;
	}

  __device__ ObjectInfo checkObject(Ray& ray);	
	__device__ Sphere::UV sphereUV(const point3& point) const;
	__device__ RGB getColor(const point3& point);
};

/**
 * @brief A plane defined by ax + by + cz + d = 0.
 */
class Plane{
public:
	float a,b,c,d; //!< a,b,c,d in ax + by + cz + d = 0.
	vec3 nor;	//!< Normal of the plane.
	point3 point; //!< A point on the plane, for calculation purposes.
	Materials mat; //!< Material properties.

	Plane(): a(0.0f), b(0.0f), c(0.0f), d(0.0f), nor(vec3()), point(point3()) 
	{
		mat.color = {1.0f,1.0f,1.0f};
	}

	Plane(float a, float b, float c, float d, RGB rgb): a(a),b(b),c(c),d(d) 
	{
		nor = vec3(a,b,c).normalize();
		point = (-d * vec3(a, b, c)) / (pow(a, 2) + pow(b, 2) + pow(c, 2));
		mat.color = rgb;
	}
	__host__ __device__ void setProperties(RGB shine,RGB tran,float ior,float roughness)
	{
		mat.shininess = shine;
		mat.trans = tran;
		mat.ior = ior;
		mat.roughness = roughness;
	}
};

/**
 * @brief A Vertex class, used only in input parsing.
 */
class Vertex{
public:
	point3 p; //!< The point for the vertex.

	Vertex(): p(point3()) {}
	Vertex(float x,float y,float z): p(point3(x,y,z)) {}
};

/**
 * @brief Class Triangle, with 3 vertices that made it up.
 */
class Triangle {
public:
	point3 p0,p1,p2; //!< Three vertices of the triangle.
	vec3 nor;  //!< Normal of the triangle.
	point3 e1,e2; //!< The e1,e2 coordinates, precomputed for Barycentric calculation.
	Materials mat; //!< Material properties.
	AABB bbox; //!< Axis-aligned bounding box for the Object class. For BVH traversal.

	Triangle()
	{
		mat.color = {1.0f, 1.0f, 1.0f};
	}

	Triangle(Vertex a, Vertex b, Vertex c, RGB rgb) 
	{
		p0 = a.p;
		p1 = b.p;
		p2 = c.p;

		mat.color = rgb;
		nor = cross(p1 - p0, p2 - p0).normalize();
		
		vec3 a1 = cross(p2 - p0, nor);
		vec3 a2 = cross(p1 - p0, nor);
		
		e1 = (1 / (dot(a1, p1 - p0))) * a1;
		e2 = (1 / (dot(a2, p2 - p0))) * a2;

		bbox = AABB(a.p, b.p, c.p);
	}

	__host__ __device__ AABB getBox() const {
		return bbox;
	}

	__host__ __device__ void setProperties(RGB shine,RGB tran,float ior,float roughness) {
		mat.shininess = shine;
		mat.trans = tran;
		mat.ior = ior;
		mat.roughness = roughness;
	}

	__device__ ObjectInfo checkObject(Ray& ray);
	__device__ RGB getColor(float b0,float b1,float b2);
};

class Sun{
public:
	vec3 dir;
	RGB color;

	Sun() 
	{
		dir = vec3(0.0f, 0.0f, 0.0f);
		color = {1.0f, 1.0f, 1.0f};
	}
	
	Sun(float x,float y,float z,RGB rgb) 
	{
		dir = vec3(x,y,z);
		color = rgb;
	}
};

class Bulb{
public:
	point3 point;
	RGB color;
	
	Bulb() 
	{
		point = point3(0.0f, 0.0f, 0.0f);
		color = {1.0f, 1.0f, 1.0f};
	}
	
	Bulb(float x, float y, float z, RGB rgb) 
	{
		point = point3(x,y,z);
		color = rgb;
	}
};

#endif
