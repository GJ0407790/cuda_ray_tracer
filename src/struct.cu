#include "config.hpp"
#include "helper.cuh"
#include "struct.cuh"

#include <math.h>

#define M_PI 3.14159265358979323846
#define EPSILON 0.001

/**
 * @brief Construct a Ray object with the xy location of the coordinates
 * @param x width location of the pixel
 * @param y height location of the pixel
 * @details Only for primary ray generation.
 */
__device__ Ray::Ray(double x, double y, curandState* state, RawConfig* config){
	auto max_dim = fmax((double)config->width, (double)config->height);
	
	double sx = (2.0f * x - config->width) / max_dim;
	double sy = (config->height - 2.0f * y) / max_dim;
	
	this->eye = config->eye;
	
	if(config->fisheye) {
		dir = (sx * config->right + sy * config->up) + sqrt(1-pow(sx,2)-pow(sy,2)) * config->forward;
	} else if (config->panorama) {
		//re-map the sx and sy
		sx = (double)x / (double) config->width;
		sy = (double)y / (double) config->height;
		//x to 360 deg, y to 180 deg
		double theta = (sx - 0.5f) * 2.0f * M_PI;
		double phi = (sy - 0.5f) * M_PI;
		//adjust to normal cylindrical projection
		dir = cos(phi) * (cos(theta) * config->forward + sin(theta) * config->right) - sin(phi) * config->up;
		dir = dir.normalize();
	//depth of field primary ray
	}else if(config->dof_focus != 0){
		double theta = randD(0, 2.0 * M_PI, state);
		double r = randD(0, config->dof_lens, state);
		double x = r * cos(theta); 
		double y = r * sin(theta);
		this->eye = this->eye + x * config->up + y * config->right;
		vec3 old_dir = config->forward + sx * config->right + sy * config->up;
		dir = (config->eye + old_dir.normalize() * config->dof_focus - this->eye) / config->dof_focus;
	}
	//regular pixel mapping
	else dir = config->forward + sx * config->right + sy * config->up;
	bounce = config->bounces;
	dir = dir.normalize();
}

__device__ Sphere::UV Sphere::sphereUV(const point3& point) const {
	vec3 s_coord = point - this->c;
	double u = (atan2(s_coord.z,s_coord.x) + M_PI) /(M_PI * 2);
	double v = acos(s_coord.y / this->r) / M_PI;
	return UV(u,v);
}

__device__ RGB Sphere::getColor(const point3& point)
{
	return mat.color;
}

__device__ RGB Triangle::getColor(double b0, double b1, double b2)
{
	return mat.color;
}

__device__ ObjectInfo Sphere::checkObject(Ray& ray){
	vec3 nor;
	RGB s_color;
	vec3 cr0 = (c - ray.eye);
	bool inside = (cr0.dot(cr0) < r * r);
	double tc = cr0.dot(ray.dir) / ray.dir.length();
	if(!inside && tc < 0) return ObjectInfo();

	vec3 d = ray.eye + (tc * ray.dir) - c;
	double d2 = pow(d.length(),2);

	if(!inside && pow(r,2) < d2) return ObjectInfo();

	//difference between t and tc
	//the two intersecting points are generated
	double t_offset = sqrt(pow(r,2) - d2) / ray.dir.length();
	double t;
	if(inside) t = tc + t_offset;
	else t = tc - t_offset;
	point3 p = t * ray.dir + ray.eye;
	s_color = this->getColor(p);
	nor = (inside) ? 1/r * (c - p) : 1/r * (p - c);
	return ObjectInfo(t,p,nor,mat); 
}

__device__ ObjectInfo Triangle::checkObject(Ray& ray)
{
	point3 intersection_point;
	RGB t_color;
	vec3 normal;
	//t is the distance the ray travels toward the triangle
	double t = dot((p0 - ray.eye),nor) / (dot(ray.dir,nor));
	
	if(t <= 0) return ObjectInfo();

	intersection_point = t * ray.dir + ray.eye;
	auto barycenter = getBarycentric(*this,intersection_point);

	double b0 = barycenter.b0;
	double b1 = barycenter.b1;
	double b2 = barycenter.b2;

	bool inside = (fmin(fmin(b0, b1), b2) >= -EPSILON);

	if(!inside && t > 0.00000001) return ObjectInfo(); //magic number, epsilon but smaller
	
	t_color = this->getColor(b0, b1, b2);
	normal = (dot(nor, ray.dir) < 0) ? nor : -nor; //determine the direction normal points to
	
	return ObjectInfo(t, intersection_point, normal, mat); 
}

// Object class function
Object::~Object() 
{
	if (obj_ptr) 
	{
		switch (obj_type) 
		{
			case ObjectType::Sphere:
				delete static_cast<Sphere*>(obj_ptr);
				break;
			case ObjectType::Triangle:
				delete static_cast<Triangle*>(obj_ptr);
				break;
			case ObjectType::BVH:
				delete static_cast<BVH*>(obj_ptr);
				break;
			default:
				break;
		}
	}
}

__host__ __device__ AABB Object::getBox() const
{
	if (obj_type == ObjectType::Sphere)
	{
		return static_cast<Sphere*>(obj_ptr)->getBox();
	}
	else if (obj_type == ObjectType::Triangle)
	{
		return static_cast<Triangle*>(obj_ptr)->getBox();
	}
	else if (obj_type == ObjectType::BVH)
	{
		return static_cast<BVH*>(obj_ptr)->getBox();
	}

	return AABB();
}

__host__ __device__ void Object::setProperties(RGB shine, RGB tran, double ior, double roughness)
{
	if (obj_type == ObjectType::Sphere)
	{
		static_cast<Sphere*>(obj_ptr)->setProperties(shine, tran, ior, roughness);
	}
	else if (obj_type == ObjectType::Triangle)
	{
		static_cast<Triangle*>(obj_ptr)->setProperties(shine, tran, ior, roughness);
	}

	// do nothing if it's not sphere or triangle
}

__device__ ObjectInfo Object::checkObject(Ray& ray)
{
	if (obj_type == ObjectType::Sphere)
	{
		return static_cast<Sphere*>(obj_ptr)->checkObject(ray);
	}
	else if (obj_type == ObjectType::Triangle)
	{
		return static_cast<Triangle*>(obj_ptr)->checkObject(ray);
	}
	else if (obj_type == ObjectType::BVH)
	{
		return static_cast<BVH*>(obj_ptr)->checkObject(ray);
	}

	return ObjectInfo();
}
