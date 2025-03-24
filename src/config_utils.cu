// config_utils.cu

#include "config_utils.cuh"
#include "parse.hpp"  // if needed for parse or stl -> raw config
#include <cstring>    // for memcpy if needed
#include <iostream>

//--------------------------------------
// 1) initRawConfigFromStl
//--------------------------------------
void initRawConfigFromStl(const StlConfig& host_stl, RawConfig& out_rc) {
	out_rc.width     = host_stl.width;
	out_rc.height    = host_stl.height;
	out_rc.color     = host_stl.color;
	out_rc.bounces   = host_stl.bounces;
	out_rc.aa        = host_stl.aa;
	out_rc.dof_focus = host_stl.dof_focus;
	out_rc.dof_lens  = host_stl.dof_lens;
	out_rc.forward   = host_stl.forward;
	out_rc.right     = host_stl.right;
	out_rc.up        = host_stl.up;
	out_rc.eye       = host_stl.eye;
	out_rc.target_up = host_stl.target_up;
	out_rc.expose    = host_stl.expose;
	out_rc.fisheye   = host_stl.fisheye;
	out_rc.panorama  = host_stl.panorama;
	out_rc.ior       = host_stl.ior;
	out_rc.rough     = host_stl.rough;
	out_rc.gi        = host_stl.gi;
	out_rc.trans     = host_stl.trans;
	out_rc.shine     = host_stl.shine;

	// For bvh_head, sun, bulbs, planes, etc., we only store them as host pointers initially
	// We'll allocate arrays on host, then device. For example:
	out_rc.num_sun = host_stl.sun.size();
	if (out_rc.num_sun > 0)
	{
		// Allocate a host array of `Sun`
		Sun** host_array = new Sun*[out_rc.num_sun];
		// Copy each pointed-to Sun from host_stl.sun[i]
		for (int i = 0; i < out_rc.num_sun; i++) 
		{
			// host_stl.sun[i] is a Sun*,
			// so we dereference it to get a Sun object
			host_array[i] = host_stl.sun[i];
		}
		out_rc.sun = host_array; // store host pointer
	}
	else
	{
		out_rc.sun = nullptr;
	}

	// 2) Bulbs
	out_rc.num_bulbs = host_stl.bulbs.size();
	if (out_rc.num_bulbs > 0) 
	{
		Bulb** host_array = new Bulb*[out_rc.num_bulbs];
		for (int i = 0; i < out_rc.num_bulbs; i++)
		{
			host_array[i] = host_stl.bulbs[i];
		}
		out_rc.bulbs = host_array;
	}
	else
	{
		out_rc.bulbs = nullptr;
	}

	// 3) Planes
	out_rc.num_planes = host_stl.planes.size();
	if (out_rc.num_planes > 0)
	{
		Plane** host_array = new Plane*[out_rc.num_planes];
		for (int i = 0; i < out_rc.num_planes; i++)
		{
			host_array[i] = host_stl.planes[i];
		}
		out_rc.planes = host_array;
	}
	else
	{
		out_rc.planes = nullptr;
	}

	// same for bulbs, planes, vertices, triangles...
	// For bvh_head:
	out_rc.bvh_head = host_stl.bvh_head ? host_stl.bvh_head : nullptr;
}

//--------------------------------------
// 2) copyRawConfigToDevice
//--------------------------------------
void copyRawConfigToDevice(RawConfig& rc) {
	if (rc.num_sun > 0 && rc.sun) 
	{
		Sun** d_sun_array = nullptr;
    cudaMalloc(&d_sun_array, rc.num_sun * sizeof(Sun*));

    for (int i = 0; i < rc.num_sun; i++)
    {
			Sun* d_sun_obj = nullptr;
			cudaMalloc(&d_sun_obj, sizeof(Sun));
			cudaMemcpy(d_sun_obj, rc.sun[i], sizeof(Sun), cudaMemcpyHostToDevice);
			cudaMemcpy(&d_sun_array[i], &d_sun_obj, sizeof(Sun*), cudaMemcpyHostToDevice);
    }

    delete[] rc.sun;
    rc.sun = d_sun_array;
	}
	
	if (rc.num_bulbs > 0 && rc.bulbs) 
	{
		Bulb** d_bulb_array = nullptr;
    cudaMalloc(&d_bulb_array, rc.num_bulbs * sizeof(Bulb*));

    for (int i = 0; i < rc.num_bulbs; i++)
    {
			Bulb* d_bulb_obj = nullptr;
			cudaMalloc(&d_bulb_obj, sizeof(Bulb));
			cudaMemcpy(d_bulb_obj, rc.bulbs[i], sizeof(Bulb), cudaMemcpyHostToDevice);
			cudaMemcpy(&d_bulb_array[i], &d_bulb_obj, sizeof(Bulb*), cudaMemcpyHostToDevice);
    }

    delete[] rc.bulbs;
    rc.bulbs = d_bulb_array;
	}

	if (rc.num_planes > 0 && rc.planes) 
	{
		Plane** d_plane_array = nullptr;
    cudaMalloc(&d_plane_array, rc.num_planes * sizeof(Plane*));

    for (int i = 0; i < rc.num_planes; i++)
    {
			Plane* d_plane_obj = nullptr;
			cudaMalloc(&d_plane_obj, sizeof(Plane));
			cudaMemcpy(d_plane_obj, rc.planes[i], sizeof(Plane), cudaMemcpyHostToDevice);
			cudaMemcpy(&d_plane_array[i], &d_plane_obj, sizeof(Plane*), cudaMemcpyHostToDevice);
    }

    delete[] rc.planes;
    rc.planes = d_plane_array;
	}

	// bvh_head
	if (rc.bvh_head) 
	{
		rc.bvh_head = deepCopyObjectToDevice(rc.bvh_head);
	}
}

//--------------------------------------
// 3) deepCopyObjectToDevice
//--------------------------------------
Object* deepCopyObjectToDevice(const Object* host_obj) 
{
	if (!host_obj) return nullptr;

	void* d_inner = nullptr;
	switch (host_obj->obj_type)
  {
		case ObjectType::Sphere: 
		{
			cudaMalloc(&d_inner, sizeof(Sphere));
			cudaMemcpy(d_inner, static_cast<Sphere*>(host_obj->obj_ptr), sizeof(Sphere), cudaMemcpyHostToDevice);
			break;
		}
		case ObjectType::Triangle: 
		{
			cudaMalloc(&d_inner, sizeof(Triangle));
			cudaMemcpy(d_inner, static_cast<Triangle*>(host_obj->obj_ptr), sizeof(Triangle), cudaMemcpyHostToDevice);
			break;
		}
		case ObjectType::BVH: 
		{
			// recursively copy BVH
			d_inner = copyBVHToDevice(static_cast<BVH*>(host_obj->obj_ptr));
			break;
		}
		default:
			return nullptr;
	}

	// 2) Create a device-side Object wrapper
	Object h_obj(host_obj->obj_type, d_inner);
	Object* d_obj = nullptr;

	cudaMalloc(&d_obj, sizeof(Object));
	cudaMemcpy(d_obj, &h_obj, sizeof(Object), cudaMemcpyHostToDevice);

	return d_obj;
}

//--------------------------------------
// 4) copyBVHToDevice
//--------------------------------------
BVH* copyBVHToDevice(BVH* host_bvh) {
	if (!host_bvh) return nullptr;

	BVH* d_bvh = nullptr;
	cudaMalloc(&d_bvh, sizeof(BVH));

	// Copy node from device->host so we can fix up pointers
	BVH temp = *host_bvh;

	// Recursively copy left child
	if (temp.left) 
	{
		Object* d_left = deepCopyObjectToDevice(temp.left);
		temp.left = d_left;
	}

	// Right child
	if (temp.right) 
	{
		Object* d_right = deepCopyObjectToDevice(temp.right);
		temp.right = d_right;
	}

	cudaMemcpy(d_bvh, &temp, sizeof(BVH), cudaMemcpyHostToDevice);
	return d_bvh;
}

//--------------------------------------
// 5) freeBVHOnDevice
//--------------------------------------
void freeBVHOnDevice(BVH* d_bvh) {
	if (!d_bvh) return;

	// Copy to host to read left/right pointers
	BVH h_bvh;
	cudaMemcpy(&h_bvh, d_bvh, sizeof(BVH), cudaMemcpyDeviceToHost);

	if (h_bvh.left) 
	{
		freeDeviceObject(h_bvh.left);
	}

	if (h_bvh.right) 
	{
		freeDeviceObject(h_bvh.right);
	}

	cudaFree(d_bvh);
}

//--------------------------------------
// 6) freeDeviceObject
//--------------------------------------
void freeDeviceObject(Object* d_obj) {
	if (!d_obj) return;

	Object h_obj;
	cudaMemcpy(&h_obj, d_obj, sizeof(Object), cudaMemcpyDeviceToHost);

	switch (h_obj.obj_type) 
	{
		case ObjectType::Sphere:
		case ObjectType::Triangle:
			// free the underlying
			if (h_obj.obj_ptr) 
			{
				cudaFree(h_obj.obj_ptr);
			}
			break;
		case ObjectType::BVH:
			freeBVHOnDevice(static_cast<BVH*>(h_obj.obj_ptr));
			break;
		default:
			break;
	}
	cudaFree(d_obj);
}

//--------------------------------------
// 7) freeRawConfigDeviceMemory
//--------------------------------------
void freeRawConfigDeviceMemory(RawConfig& rc) {
	// free sun
	if (rc.sun) 
	{
		for (int i = 0; i < rc.num_sun; i++)
		{
			cudaFree(rc.sun[i]);
		}
		cudaFree(rc.sun);
		rc.sun = nullptr;
	}

	// free bulbs
	if (rc.bulbs) 
	{
		for (int i = 0; i < rc.num_bulbs; i++)
		{
			cudaFree(rc.bulbs[i]);
		}
		cudaFree(rc.bulbs);
		rc.bulbs = nullptr;
	}

	// free planes
	if (rc.planes) 
	{
		for (int i = 0; i < rc.num_planes; i++)
		{
			cudaFree(rc.planes[i]);
		}
		cudaFree(rc.planes);
		rc.planes = nullptr;
	}

	// free bvh_head
	if (rc.bvh_head) 
	{
		freeDeviceObject(rc.bvh_head);
		rc.bvh_head = nullptr;
	}
}

void freeStlConfig(StlConfig& stl) 
{
	// Free the BVH head pointer if it exists
	if (stl.bvh_head) 
	{
		freeObject(stl.bvh_head); 
		stl.bvh_head = nullptr;
	}

	for (auto obj : stl.objects) 
	{
		if (obj) 
		{
			freeObject(obj);
		}
	}
	stl.objects.clear();

	for (auto sun : stl.sun)
	{
		if (sun)
		{
			delete sun;
		}
	}
	stl.sun.clear();

	for (auto bulb : stl.bulbs)
	{
		if (bulb)
		{
			delete bulb;
		}
	}
	stl.bulbs.clear();

	for (auto plane : stl.planes)
	{
		if (plane)
		{
			delete plane;
		}
	}
	stl.planes.clear();

	for (auto v : stl.vertices)
	{
		if (v)
		{
			delete v;
		}
	}
	stl.vertices.clear();
}

void freeObject(Object* obj) 
{
	if (obj->obj_ptr) 
	{
		switch (obj->obj_type) 
		{
			case ObjectType::Sphere:
				delete static_cast<Sphere*>(obj->obj_ptr);
				break;
			case ObjectType::Triangle:
				delete static_cast<Triangle*>(obj->obj_ptr);
				break;
			case ObjectType::BVH:
				freeBvh(static_cast<BVH*>(obj->obj_ptr));
				break;
			default:
				break;
		}
	}

	delete obj;
}

void freeBvh(BVH* bvh) 
{
	if (bvh->left && bvh->left->obj_type == ObjectType::BVH)
	{
		freeObject(bvh->left);
	}
	
	if (bvh->right && bvh->right->obj_type == ObjectType::BVH)
	{
		freeObject(bvh->right);
	}
}
