#ifndef OBJECT_H
#define OBJECT_H

#include "struct.cuh"
#include "config.hpp"
#include "libpng.h"

#include <algorithm>
#include <memory>

/**
 * @brief Material related variables for objects in the scene.
 * 		  
 */
class Materials{
public:
	RGB color; //!< The color, in RGB format, of the material
	RGB shininess; //!< The shininess of the object, which is related to reflection. 
	RGB trans; 	//! < The transparency of the object, which is related to refraction. 
	double ior = 1.458; //!< Index of refraction. Default to 1.458.
	double roughness = 0; //!< Roughness of the object, Default to zero(none).
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
	__host__ __device__ Materials(RGB color,RGB shininess,RGB trans,double ior,double roughness):
		color(color),shininess(shininess),trans(trans),ior(ior),roughness(roughness) {}
};

/**
 * @brief ObjectInfo is a class that passes the parameters of a ray hit, and the
 * 		details of the object that was hit.
 */
class ObjectInfo{
public:
	bool isHit = false; //!< Signals if there was a hit. Defaults to false.
	double distance; //!< The distance from the ray origin to the object.
	point3 i_point;  //!< The intersection point.
	vec3 normal; //!< The normal at the point of intersection.
	Materials mat; //!< Material properties.
	
	__device__ ObjectInfo(): distance(-1.0),i_point(point3()),normal(vec3()),mat(Materials()) {}
  __device__ ObjectInfo(double distance, point3 i_point, vec3 normal,Materials mat)
							: isHit(true), distance(distance), i_point(i_point), 
							  normal(normal), mat(mat) {}
};

class RawConfig;

/**
 * @brief Class Ray, consists of a eye and direction
 */
class Ray{
public:
	point3 eye;
	vec3 dir;
	int bounce;
	__device__ Ray() : bounce(0) {}
	__device__ Ray(double x, double y, curandState* state, RawConfig* config);
	__device__ Ray(point3 eye,vec3 dir,int bounce): eye(eye), dir(dir.normalize()),bounce(bounce){}
};

class AABB{
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
		Interval x(x_min, x_max);
		Interval y(y_min, y_max);
		Interval z(z_min, z_max);
		
		//expand if the interval is too small
		if(x.size() < 0.01) x = x.expand(0.01);
		if(y.size() < 0.01) y = y.expand(0.01);
		if(z.size() < 0.01) z = z.expand(0.01);
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


	__host__ __device__ bool hit(const Ray& r) const 
	{
		const point3& ray_eye = r.eye;
    const vec3&   ray_dir = r.dir;

    double t_min = -INFINITY;
    double t_max =  INFINITY;

    for (int axis = 0; axis < 3; axis++) {
			const Interval& ax = getAxis(axis);
			
			if (ax.min > ax.max) return false;

			const double adinv = 1.0 / ray_dir[axis];

			double t0 = (ax.min - ray_eye[axis]) * adinv;
			double t1 = (ax.max - ray_eye[axis]) * adinv;

			// Manual swap
			if (t0 > t1) {
				double temp = t0;
				t0 = t1;
				t1 = temp;
			}

			t_min = fmax(t_min, t0);
			t_max = fmin(t_max, t1);

			if (t_max <= t_min) return false;
    }

    return true;
	}
};


enum class ObjectType {
	Sphere,
	Triangle,
	BVH,
	None
};

/**
 * @brief The object class, both BVH nodes and objects in the scene.
 */
class Object {
public:
	Object() {}
	
	Object(ObjectType type, void* ptr)
	{
		obj_type = type;
		obj_ptr = ptr;
	}

	~Object() {};

	__host__ __device__ AABB getBox() const;
	__host__ __device__ void setProperties(RGB shine, RGB tran, double ior, double roughness);
	__device__ ObjectInfo checkObject(Ray& ray);

public:
	ObjectType obj_type = ObjectType::None;
	void* obj_ptr = nullptr;
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
		double u;
		double v;

		__device__ UV(double u_val, double v_val)
								: u(u_val), v(v_val) {}
	};

public:
	point3 c; //!< Center point of the sphere.
	double r; //!< Radius of the sphere.
	Materials mat; //!< Material properties.
	AABB bbox; //!< Axis-aligned bounding box for the Object class. For BVH traversal.

	/**
	 * @brief Construct a new Sphere object with no inputs
	 */
	Sphere(): r(0.0) {
		c = point3(0.0,0.0,0.0);
		mat.color = {1.0f,1.0f,1.0f};
	}
	/**
	 * @brief Construct a new Sphere object, with color inputs.
	 */
	Sphere(double x,double y,double z,double r,RGB rgb): r(r) {
		c = point3(x,y,z);
		mat.color = rgb;
		auto rvec = vec3(r,r,r);
    bbox = AABB(c - rvec,c + rvec);
	}
	
	/**
	 * @brief Construct a new Sphere object, with texture inputs.
	 */
	Sphere(double x,double y,double z,double r,std::string text): r(r) {
		c = point3(x,y,z);
		auto rvec = vec3(r,r,r);
    bbox = AABB(c - rvec,c + rvec);
	}

	__host__ __device__ void setProperties(RGB shine, RGB tran, double ior, double roughness) {
		mat.shininess = shine;
		mat.trans = tran;
		mat.ior = ior;
		mat.roughness = roughness;
	}

	__host__ __device__ AABB getBox() const {
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
	double a,b,c,d; //!< a,b,c,d in ax + by + cz + d = 0.
	vec3 nor;	//!< Normal of the plane.
	point3 point; //!< A point on the plane, for calculation purposes.
	Materials mat; //!< Material properties.

	Plane(): a(0.0),b(0.0),c(0.0),d(0.0),nor(vec3()),point(point3()) {mat.color = {1.0f,1.0f,1.0f};}
	Plane(double a,double b,double c,double d,RGB rgb): a(a),b(b),c(c),d(d) {
		nor = vec3(a,b,c).normalize();
		point = (-d * vec3(a,b,c)) / (pow(a,2) + pow(b,2) + pow(c,2));
		mat.color = rgb;
	}
	__host__ __device__ void setProperties(RGB shine,RGB tran,double ior,double roughness){
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
	Texcoord tex; //<! The texture coordinate, if provided.
	Vertex(): p(point3()) {}
	Vertex(double x,double y,double z): p(point3(x,y,z)) {}
	Vertex(double x,double y,double z,Texcoord tex): p(point3(x,y,z)),tex(tex){}
};

/**
 * @brief Class Triangle, with 3 vertices that made it up.
 */
class Triangle {
public:
	point3 p0,p1,p2; //!< Three vertices of the triangle.
	Texcoord tex0,tex1,tex2; //!<texture coordinates for the vertices,if provided.
	vec3 nor;  //!< Normal of the triangle.
	point3 e1,e2; //!< The e1,e2 coordinates, precomputed for Barycentric calculation.
	Materials mat; //!< Material properties.
	AABB bbox; //!< Axis-aligned bounding box for the Object class. For BVH traversal.

	Triangle(): p0(point3()),p1(point3()),p2(point3()),
				nor(vec3()),e1(point3()),e2(point3()){mat.color = {1.0f,1.0f,1.0f};}

	Triangle(Vertex a,Vertex b,Vertex c,RGB rgb) {
		p0 = a.p;p1 = b.p;p2 = c.p;
		tex0 = a.tex;tex1 = b.tex;tex2 = c.tex;
		mat.color = rgb;
		nor = cross(p1-p0,p2-p0).normalize();
		vec3 a1 = cross(p2-p0,nor);
		vec3 a2 = cross(p1-p0,nor);
		e1 = (1/(dot(a1,p1-p0))) * a1;
		e2 = (1/(dot(a2,p2-p0))) * a2;
		bbox = AABB(a.p,b.p,c.p);
	}

	Triangle(Vertex a,Vertex b,Vertex c,std::string text) {
		p0 = a.p;p1 = b.p;p2 = c.p;
		tex0 = a.tex;tex1 = b.tex;tex2 = c.tex;
		nor = cross(p1-p0,p2-p0).normalize();
		vec3 a1 = cross(p2-p0,nor);
		vec3 a2 = cross(p1-p0,nor);
		e1 = (1/(dot(a1,p1-p0))) * a1;
		e2 = (1/(dot(a2,p2-p0))) * a2;
		bbox = AABB(a.p,b.p,c.p);
	}

	__host__ __device__ AABB getBox() const {
		return bbox;
	}

	__host__ __device__ void setProperties(RGB shine,RGB tran,double ior,double roughness) {
		mat.shininess = shine;
		mat.trans = tran;
		mat.ior = ior;
		mat.roughness = roughness;
	}

	__device__ ObjectInfo checkObject(Ray& ray);
	__device__ RGB getColor(double b0,double b1,double b2);
};

class Sun{
public:
	vec3 dir;
	RGB color;
	Sun() {
		dir = vec3(0.0,0.0,0.0);
		color = {1.0f,1.0f,1.0f};
	}
	
	Sun(double x,double y,double z,RGB rgb) {
		dir = vec3(x,y,z);
		color = rgb;
	}
};

class Bulb{
public:
	point3 point;
	RGB color;
	
	Bulb() {
		point = point3(0.0,0.0,0.0);
		color = {1.0f,1.0f,1.0f};
	}
	
	Bulb(double x,double y,double z,RGB rgb) {
		point = point3(x,y,z);
		color = rgb;
	}
};


class BVH {
public:
	BVH() {}

	BVH(std::vector<Object*>& objs, int start, int end, int axis)
	{
		auto comp = (axis == 0) ? box_x_compare
							: (axis == 1) ? box_y_compare
														: box_z_compare;

		int span = end - start;

		if(span == 0)
		{
			return;
		}
		else if(span == 1)
		{
			left = objs[start];
			right = objs[start];
		}
		else if(span == 2)
		{
			left = objs[start];
			right = objs[start+1];
		}
		else
		{
			std::sort(std::begin(objs) + start,std::begin(objs) + end,comp);
			int mid = start + span/2;
			BVH* left_bvh = new BVH(objs, start, mid, (axis+1)%3);
      BVH* right_bvh = new BVH(objs, mid, end, (axis+1)%3);

			left = new Object(ObjectType::BVH, left_bvh);
			right = new Object(ObjectType::BVH, right_bvh);
		}

		bbox = AABB(left->getBox(), right->getBox());
	}

	// need to recursively delete the allocated object
	~BVH() {}
    
	__host__ __device__ AABB getBox() const 
	{
		return bbox;
	}

  __device__ ObjectInfo checkObject(Ray& ray) 
	{
		//hit nothing, return nothing
		if (!bbox.hit(ray))
				return ObjectInfo();

		ObjectInfo leftInfo = left->checkObject(ray);
		ObjectInfo rightInfo = right->checkObject(ray);

		if(leftInfo.isHit && !rightInfo.isHit) 
		{
			return leftInfo;
		}
		else if(!leftInfo.isHit && rightInfo.isHit) 
		{
			return rightInfo;
		}
		else if(!leftInfo.isHit && !rightInfo.isHit)
		{
			return ObjectInfo();
		}
		else
		{
			if(leftInfo.distance < rightInfo.distance) return leftInfo;
			else return rightInfo;
		}
	}

	static bool box_compare(const Object* a, const Object* b, int axis_index) 
	{
		auto a_axis_interval = a->getBox().getAxis(axis_index);
		auto b_axis_interval = b->getBox().getAxis(axis_index);
		return a_axis_interval.min < b_axis_interval.min;
  }

	static bool box_x_compare (const Object* a, const Object* b) 
	{
		return box_compare(a, b, 0);
	}

	static bool box_y_compare (const Object* a, const Object* b) 
	{
		return box_compare(a, b, 1);
	}

	static bool box_z_compare (const Object* a, const Object* b) 
	{
		return box_compare(a, b, 2);
	} 
   
public:
	Object* left = nullptr;
	Object* right = nullptr;
	AABB bbox;
};

#endif
