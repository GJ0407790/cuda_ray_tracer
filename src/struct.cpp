/**
 * @file struct.cpp
 * @author Jin (jinj2@illinois.edu)
 * @brief 
 * @version 0.1
 * @date 2024-11-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "../include/all.hpp"
#include <random>
using std::sqrt;
using std::pow;
using std::cos;
using std::sin;

#define M_PI 3.14159265358979323846
#define EPSILON 0.001
/**
 * @brief Construct a Ray object with the xy location of the coordinates
 * @param x width location of the pixel
 * @param y height location of the pixel
 * @details Only for primary ray generation.
 */
Ray::Ray(double x,double y){
	double sx = (2.0f * x - width) / std::max(width,height);
	double sy = (height - 2.0f * y) / std::max(width,height);
	this->eye = ::eye;
	if(fisheye) dir = (sx * right + sy * up) + sqrt(1-pow(sx,2)-pow(sy,2)) * forward;
	else if(panorama){
		//re-map the sx and sy
		sx = (double)x / (double)width;
		sy = (double)y / (double)height;
		//x to 360 deg, y to 180 deg
		double theta = (sx - 0.5f) * 2.0f * M_PI;
		double phi = (sy - 0.5f) * M_PI;
		//adjust to normal cylindrical projection
		dir = cos(phi) * (cos(theta) * forward + sin(theta) * right) - sin(phi) * up;
		dir = dir.normalize();
	//depth of field primary ray
	}else if(dof_focus != 0){
		double theta = randD(0,2*M_PI);
		double r = randD(0,dof_lens);
		double x = r * std::cos(theta); double y = r * std::sin(theta);
		this->eye = this->eye + x * up + y * right;
		vec3 old_dir = forward + sx * right + sy * up;
		dir = (::eye + old_dir.normalize() * dof_focus - this->eye) / dof_focus;
	}
	//regular pixel mapping
	else dir = forward + sx * right + sy * up;
	bounce = bounces;
	dir = dir.normalize();
}

std::tuple<double,double> Sphere::sphereUV(const point3& point) const {
	vec3 s_coord = point - this->c;
	double u = (std::atan2(s_coord.z,s_coord.x) + M_PI) /(M_PI * 2);
	double v = std::acos(s_coord.y / this->r) / M_PI;
	return std::make_tuple(u,v);
}

RGB Sphere::getColor(const point3& point){
	if(texture.empty()){
		return this->color;
	}else{
		auto[u,v] = this->sphereUV(point);
		int x = (1-u) * this->texture.width();
		int y = v * this->texture.height();
		double r = this->texture[y][x].r;
		double g = this->texture[y][x].g;
		double b = this->texture[y][x].b;
		return {sRGBtoRGB(r),sRGBtoRGB(g),sRGBtoRGB(b)};
	}
}

RGB Triangle::getColor(double b0,double b1,double b2){
	if(texture.empty()){
		return this->color;
	}else{     
		auto [u0,v0] = this->tex0;
		auto [u1,v1] = this->tex1;
		auto [u2,v2] = this->tex2;
		unsigned int x = (u0 * b0 + u1 * b1 + u2 * b2) * this->texture.width();
		unsigned int y = (v0 * b0 + v1 * b1 + v2 * b2) * this->texture.height();
		x = std::min(x, this->texture.width() - 1);
		y = std::min(y, this->texture.height() - 1);
		double r = this->texture[y][x].r;
		double g = this->texture[y][x].g;
		double b = this->texture[y][x].b;
		return {sRGBtoRGB(r),sRGBtoRGB(g),sRGBtoRGB(b)};
	}
}

ObjectInfo Sphere::checkObject(Ray& ray){
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
	double t_offset = std::sqrt(pow(r,2) - d2) / ray.dir.length();
	double t;
	if(inside) t = tc + t_offset;
	else t = tc - t_offset;
	point3 p = t * ray.dir + ray.eye;
	s_color = this->getColor(p);
	nor = (inside) ? 1/r * (c - p) : 1/r * (p - c);
	return ObjectInfo(t,p,nor,s_color,shininess,trans,ior,roughness); 
}

ObjectInfo Triangle::checkObject(Ray& ray){
	point3 intersection_point;
	RGB t_color;
	vec3 normal;
	//t is the distance the ray travels toward the triangle
	double t = dot((p0 - ray.eye),nor) / (dot(ray.dir,nor));
	if(t <= 0) return ObjectInfo();

	intersection_point = t * ray.dir + ray.eye;
	auto [b0,b1,b2] = getBarycentric(*this,intersection_point);
	bool inside = (std::min({b0, b1, b2}) >= -EPSILON);
	if(!inside && t > 0.00000001) return ObjectInfo(); //magic number, epsilon but smaller
	t_color = this->getColor(b0,b1,b2);
	normal = (dot(nor,ray.dir) < 0) ? nor : -nor; //determine the direction normal points to
	return ObjectInfo(t,intersection_point,normal,t_color,shininess,trans,ior,roughness); 
}
