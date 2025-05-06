#include "config.hpp"
#include "helper.cuh"
#include "struct.cuh"

#include <math.h>

#define PI 3.14159265358979323846f
#define EPSILON 0.001f

/**
 * @brief Construct a Ray object with the xy location of the coordinates
 * @param x width location of the pixel
 * @param y height location of the pixel
 * @details Only for primary ray generation.
 */
__device__ Ray::Ray(float x, float y, curandState* state, RawConfig* config){
	auto max_dim = fmax((float)config->width, (float)config->height);
	
	float sx = (2.0f * x - config->width) / max_dim;
	float sy = (config->height - 2.0f * y) / max_dim;
	
	this->eye = config->eye;
	
	if(config->fisheye) 
	{
		dir = (sx * config->right + sy * config->up) 
					+ sqrt(1.0f - pow(sx, 2) - pow(sy,2)) * config->forward;
	} 
	else if (config->panorama) 
	{
		//re-map the sx and sy
		sx = (float)x / (float) config->width;
		sy = (float)y / (float) config->height;
		//x to 360 deg, y to 180 deg
		float theta = (sx - 0.5f) * 2.0f * PI;
		float phi = (sy - 0.5f) * PI;
		//adjust to normal cylindrical projection
		dir = cos(phi) * (cos(theta) * config->forward + sin(theta) * config->right) - sin(phi) * config->up;
		dir = dir.normalize();
	//depth of field primary ray
	}
	else if(config->dof_focus != 0.0f)
	{
		float theta = randD(0.0f, 2.0f * PI, state);
		float r = randD(0.0f, config->dof_lens, state);
		float x = r * cos(theta); 
		float y = r * sin(theta);

		this->eye = this->eye + x * config->up + y * config->right;
		
		vec3 old_dir = config->forward + sx * config->right + sy * config->up;
		dir = (config->eye + old_dir.normalize() * config->dof_focus - this->eye) / config->dof_focus;
	}
	//regular pixel mapping
	else 
	{
		dir = config->forward + sx * config->right + sy * config->up;
	}

	bounce = config->bounces;
	dir = dir.normalize();
}

__device__ Sphere::UV Sphere::sphereUV(const point3& point) const {
	vec3 s_coord = point - this->c;

	float u = (atan2(s_coord.z,s_coord.x) + PI) /(PI * 2.0f);
	float v = acos(s_coord.y / this->r) / PI;

	return UV(u,v);
}

__device__ RGB Sphere::getColor(const point3& point)
{
	return mat.color;
}

__device__ RGB Triangle::getColor(float b0, float b1, float b2)
{
	return mat.color;
}

__device__ ObjectInfo Sphere::checkObject(Ray& ray)
{
	vec3 nor;
	RGB s_color;
	vec3 cr0 = (c - ray.eye);
	bool inside = (cr0.dot(cr0) < r * r);

	float tc = cr0.dot(ray.dir) / ray.dir.length();
	
	if(!inside && tc < 0.0f) return ObjectInfo();

	vec3 d = ray.eye + (tc * ray.dir) - c;
	float d2 = pow(d.length(), 2);

	if(!inside && pow(r, 2) < d2) return ObjectInfo();

	//difference between t and tc
	//the two intersecting points are generated
	float t_offset = sqrt(pow(r, 2) - d2) / ray.dir.length();
	float t;

	if(inside) t = tc + t_offset;
	else t = tc - t_offset;
	
	point3 p = t * ray.dir + ray.eye;
	s_color = this->getColor(p);
	
	nor = (inside) ? 1.0f/r * (c - p) : 1.0f/r * (p - c);
	
	return ObjectInfo(t,p,nor,mat); 
}

__device__ ObjectInfo Triangle::checkObject(Ray& ray)
{
	point3 intersection_point;
	RGB t_color;
	vec3 normal;
	//t is the distance the ray travels toward the triangle
	float t = dot((p0 - ray.eye),nor) / (dot(ray.dir,nor));
	
	if(t <= 1e-6f) return ObjectInfo();

	intersection_point = t * ray.dir + ray.eye;
	auto barycenter = getBarycentric(*this, intersection_point);

	float b0 = barycenter.b0;
	float b1 = barycenter.b1;
	float b2 = barycenter.b2;

	bool inside = (b0 >= -EPSILON) && (b1 >= -EPSILON) && (b2 >= -EPSILON);

	if(!inside && t > 1e-8f) //magic number, epsilon but smaller
	{
		return ObjectInfo();
	}
	
	t_color = this->getColor(b0, b1, b2);
	normal = (dot(nor, ray.dir) < 1e-6f) ? nor : -nor; //determine the direction normal points to
	
	return ObjectInfo(t, intersection_point, normal, mat); 
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

__host__ __device__ void Object::setProperties(RGB shine, RGB tran, float ior, float roughness)
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
