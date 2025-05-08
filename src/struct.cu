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

__device__ ObjectInfo checkSphereIntersectionSoA(
	const Ray& ray,
	unsigned int sphere_idx,
	const SphereDataSoA& spheres_soa,
	const RawConfig* config // Unused for now, but good for consistency
) 
{
	point3 s_center = spheres_soa.c[sphere_idx];
	float s_radius = spheres_soa.r[sphere_idx];
	// Materials s_mat = spheres_soa.mat[sphere_idx]; // Load full material struct

	vec3 nor;
	// RGB s_color_val; // Not needed if we pass Materials struct

	vec3 cr0 = (s_center - ray.eye);
	bool inside = (cr0.dot(cr0) < s_radius * s_radius);

	float tc = cr0.dot(ray.dir); // Removed / ray.dir.length() as ray.dir should be normalized

	if(!inside && tc < 0.0f) return ObjectInfo(); // No hit

	vec3 d_vec = ray.eye + (tc * ray.dir) - s_center; // d_vec instead of d to avoid redefinition if EPSILON is d
	float d2 = d_vec.dot(d_vec); // Use dot product for squared length

	if(!inside && (s_radius * s_radius) < d2) return ObjectInfo(); // No hit

	float t_offset = sqrt((s_radius * s_radius) - d2); // Removed / ray.dir.length()

	float t_intersect;
	if(inside) 
	{
		t_intersect = tc + t_offset;
	} 
	else 
	{
		t_intersect = tc - t_offset;
	}

	point3 p_intersect = t_intersect * ray.dir + ray.eye;
	// s_color_val = s_mat.color; // Color is part of s_mat

	nor = (inside) ? (s_center - p_intersect) : (p_intersect - s_center);
	nor = nor.normalize(); // Ensure normal is unit length (division by r was implicit normalize)

	return ObjectInfo(t_intersect, p_intersect, nor, spheres_soa.mat[sphere_idx]);
}

__device__ ObjectInfo checkTriangleIntersectionSoA(
	const Ray& ray,
	unsigned int triangle_idx,
	const TriangleDataSoA& triangles_soa,
	const RawConfig* config) 
{
	point3 t_p0 = triangles_soa.p0[triangle_idx];
	vec3 t_nor_precomputed = triangles_soa.nor[triangle_idx];

	point3 intersection_point;
	vec3 final_normal;

	float denominator = dot(ray.dir, t_nor_precomputed);

	if (fabsf(denominator) < 1e-9f) // Ray is parallel or grazing
	{ 
		return ObjectInfo(); 
	}

	float t_intersect = dot((t_p0 - ray.eye), t_nor_precomputed) / denominator;

	if (t_intersect <= EPSILON) // Intersection is behind or too close
	{ 
		return ObjectInfo();
	} 

	intersection_point = t_intersect * ray.dir + ray.eye;

	// INLINED Barycentric check using precomputed e1, e2 from SoA
	point3 t_e1_bary = triangles_soa.e1[triangle_idx];
	point3 t_e2_bary = triangles_soa.e2[triangle_idx];

	float b1_val = dot(t_e1_bary, intersection_point - t_p0);
	float b2_val = dot(t_e2_bary, intersection_point - t_p0);
	float b0_val = 1.0f - b1_val - b2_val;

	// Check if point is within triangle bounds
	// A common check is b0, b1, b2 all >= 0.
	// Sometimes a small epsilon is used for robustness at edges.
	bool inside = (b0_val >= -EPSILON) && 
								(b1_val >= -EPSILON) && 
								(b2_val >= -EPSILON);

	if(!inside) 
	{
		return ObjectInfo(); // Not within triangle
	}

	// Determine the direction normal points to
	final_normal = (denominator < 0.0f) ? t_nor_precomputed : -t_nor_precomputed; 

	return ObjectInfo(t_intersect, intersection_point, final_normal, triangles_soa.mat[triangle_idx]);
}


