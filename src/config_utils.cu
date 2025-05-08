#include "config_utils.cuh"
#include <cstring>    // For memcpy if needed
#include <iostream>   // For error messages

// Initializes a host-side RawConfig structure based on a host-side StlConfig.
// Device pointers in out_rc_host_mirror will be nullptr.
void initRawConfigFromStl(const StlConfig& host_stl, RawConfig& out_rc_host_mirror) {
	// Copy simple members
	out_rc_host_mirror.width = host_stl.width;
	out_rc_host_mirror.height = host_stl.height;
	out_rc_host_mirror.color = host_stl.color;
	out_rc_host_mirror.bounces = host_stl.bounces;
	out_rc_host_mirror.aa = host_stl.aa;
	out_rc_host_mirror.dof_focus = host_stl.dof_focus;
	out_rc_host_mirror.dof_lens = host_stl.dof_lens;
	out_rc_host_mirror.forward = host_stl.forward;
	out_rc_host_mirror.right = host_stl.right;
	out_rc_host_mirror.up = host_stl.up;
	out_rc_host_mirror.eye = host_stl.eye;
	out_rc_host_mirror.target_up = host_stl.target_up;
	out_rc_host_mirror.expose = host_stl.expose;
	out_rc_host_mirror.fisheye = host_stl.fisheye;
	out_rc_host_mirror.panorama = host_stl.panorama;
	out_rc_host_mirror.ior = host_stl.ior;
	out_rc_host_mirror.rough = host_stl.rough;
	out_rc_host_mirror.gi = host_stl.gi;
	out_rc_host_mirror.trans = host_stl.trans;
	out_rc_host_mirror.shine = host_stl.shine;

	// Initialize counts for SoA data
	out_rc_host_mirror.num_spheres = host_stl.host_spheres_data.size();
	out_rc_host_mirror.num_triangles = host_stl.host_triangles_data.size();
	out_rc_host_mirror.num_total_primitives = host_stl.host_primitive_references.size();
	
	out_rc_host_mirror.scene_min_corner = make_float3(
		host_stl.scene_bounds_host.x.min,
		host_stl.scene_bounds_host.y.min,
		host_stl.scene_bounds_host.z.min
	);
	out_rc_host_mirror.scene_max_corner = make_float3(
		host_stl.scene_bounds_host.x.max,
		host_stl.scene_bounds_host.y.max,
		host_stl.scene_bounds_host.z.max
	);

	out_rc_host_mirror.num_sun = host_stl.sun_data.size();
	out_rc_host_mirror.num_bulbs = host_stl.bulb_data.size();
	out_rc_host_mirror.num_planes = host_stl.plane_data.size();

	out_rc_host_mirror.num_lbvh_nodes = 0;

	// Initialize device pointers to nullptr
	out_rc_host_mirror.d_all_spheres = nullptr;
	out_rc_host_mirror.d_all_triangles = nullptr;
	out_rc_host_mirror.d_primitive_references = nullptr;
	out_rc_host_mirror.d_morton_codes = nullptr;

	out_rc_host_mirror.d_all_suns = nullptr; 
	out_rc_host_mirror.d_all_bulbs = nullptr; 
	out_rc_host_mirror.d_all_planes = nullptr;

	out_rc_host_mirror.d_lbvh_nodes = nullptr;
}

// Allocates device memory and copies data from StlConfig to the device.
// Fills the device pointer members (d_all_spheres, sun (as d_all_suns), etc.) in rc_with_device_ptrs.
void copyConfigDataToDevice(const StlConfig& host_stl, RawConfig& rc_with_device_ptrs) {
	// --- Copy SoA Primitive Data ---
	if (rc_with_device_ptrs.num_spheres > 0) 
	{
		cudaError_t err = cudaMalloc(&rc_with_device_ptrs.d_all_spheres, rc_with_device_ptrs.num_spheres * sizeof(Sphere));
		if (err != cudaSuccess) { std::cerr << "CUDA Malloc failed for d_all_spheres: " << cudaGetErrorString(err) << std::endl; return; }
		err = cudaMemcpy(rc_with_device_ptrs.d_all_spheres, host_stl.host_spheres_data.data(), rc_with_device_ptrs.num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { std::cerr << "CUDA Memcpy failed for d_all_spheres: " << cudaGetErrorString(err) << std::endl; return; }
	}

	if (rc_with_device_ptrs.num_triangles > 0) 
	{
		cudaError_t err = cudaMalloc(&rc_with_device_ptrs.d_all_triangles, rc_with_device_ptrs.num_triangles * sizeof(Triangle));
		if (err != cudaSuccess) { std::cerr << "CUDA Malloc failed for d_all_triangles: " << cudaGetErrorString(err) << std::endl; return; }
		err = cudaMemcpy(rc_with_device_ptrs.d_all_triangles, host_stl.host_triangles_data.data(), rc_with_device_ptrs.num_triangles * sizeof(Triangle), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { std::cerr << "CUDA Memcpy failed for d_all_triangles: " << cudaGetErrorString(err) << std::endl; return; }
	}

	if (rc_with_device_ptrs.num_total_primitives > 0) 
	{
		cudaError_t err = cudaMalloc(&rc_with_device_ptrs.d_primitive_references, rc_with_device_ptrs.num_total_primitives * sizeof(PrimitiveReference));
		if (err != cudaSuccess) { std::cerr << "CUDA Malloc failed for d_primitive_references: " << cudaGetErrorString(err) << std::endl; return; }
		err = cudaMemcpy(rc_with_device_ptrs.d_primitive_references, host_stl.host_primitive_references.data(), rc_with_device_ptrs.num_total_primitives * sizeof(PrimitiveReference), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { std::cerr << "CUDA Memcpy failed for d_primitive_references: " << cudaGetErrorString(err) << std::endl; return; }

		err = cudaMalloc(&rc_with_device_ptrs.d_morton_codes, rc_with_device_ptrs.num_total_primitives * sizeof(unsigned int));
		if (err != cudaSuccess) { std::cerr << "CUDA Malloc failed for d_morton_codes: " << cudaGetErrorString(err) << std::endl; return; }
		// Morton codes are generated on device, so no cudaMemcpy for their content here.
	}

	// --- Copy SoA Lights, Planes, etc.
	if (rc_with_device_ptrs.num_sun > 0) 
	{
		cudaError_t err = cudaMalloc(&rc_with_device_ptrs.d_all_suns, rc_with_device_ptrs.num_sun * sizeof(Sun)); // Assuming RawConfig.sun is Sun*
		if (err != cudaSuccess) { std::cerr << "CUDA Malloc failed for sun data: " << cudaGetErrorString(err) << std::endl; return; }
		err = cudaMemcpy(rc_with_device_ptrs.d_all_suns, host_stl.sun_data.data(), rc_with_device_ptrs.num_sun * sizeof(Sun), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { std::cerr << "CUDA Memcpy failed for sun data: " << cudaGetErrorString(err) << std::endl; return; }
	}

	if (rc_with_device_ptrs.num_bulbs > 0) 
	{
		cudaError_t err = cudaMalloc(&rc_with_device_ptrs.d_all_bulbs, rc_with_device_ptrs.num_bulbs * sizeof(Bulb)); // Assuming RawConfig.bulbs is Bulb*
		if (err != cudaSuccess) { std::cerr << "CUDA Malloc failed for bulb data: " << cudaGetErrorString(err) << std::endl; return; }
		err = cudaMemcpy(rc_with_device_ptrs.d_all_bulbs, host_stl.bulb_data.data(), rc_with_device_ptrs.num_bulbs * sizeof(Bulb), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { std::cerr << "CUDA Memcpy failed for bulb data: " << cudaGetErrorString(err) << std::endl; return; }
	}

	if (rc_with_device_ptrs.num_planes > 0) 
	{
		cudaError_t err = cudaMalloc(&rc_with_device_ptrs.d_all_planes, rc_with_device_ptrs.num_planes * sizeof(Plane)); // Assuming RawConfig.planes is Plane*
		if (err != cudaSuccess) { std::cerr << "CUDA Malloc failed for plane data: " << cudaGetErrorString(err) << std::endl; return; }
		err = cudaMemcpy(rc_with_device_ptrs.d_all_planes, host_stl.plane_data.data(), rc_with_device_ptrs.num_planes * sizeof(Plane), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { std::cerr << "CUDA Memcpy failed for plane data: " << cudaGetErrorString(err) << std::endl; return; }
	}
}

// Frees all device memory pointed to by members of rc_with_device_ptrs.
void freeRawConfigDeviceMemory(RawConfig& rc_with_device_ptrs) {
	cudaFree(rc_with_device_ptrs.d_all_spheres);
	rc_with_device_ptrs.d_all_spheres = nullptr;
	cudaFree(rc_with_device_ptrs.d_all_triangles);
	rc_with_device_ptrs.d_all_triangles = nullptr;
	cudaFree(rc_with_device_ptrs.d_primitive_references);
	rc_with_device_ptrs.d_primitive_references = nullptr;
	cudaFree(rc_with_device_ptrs.d_morton_codes);
	rc_with_device_ptrs.d_morton_codes = nullptr;

	// Free SoA Lights, Planes, etc.
	cudaFree(rc_with_device_ptrs.d_all_suns);
	rc_with_device_ptrs.d_all_suns = nullptr;
	cudaFree(rc_with_device_ptrs.d_all_bulbs);
	rc_with_device_ptrs.d_all_bulbs = nullptr;
	cudaFree(rc_with_device_ptrs.d_all_planes);
	rc_with_device_ptrs.d_all_planes = nullptr;

	// For LBVH
	cudaFree(rc_with_device_ptrs.d_lbvh_nodes);
	rc_with_device_ptrs.d_lbvh_nodes = nullptr;
}

