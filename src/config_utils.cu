#include "config_utils.cuh"
#include <cstring>    // For memcpy if needed
#include <iostream>   // For error messages

#define CUDA_CHECK_CONFIG_UTILS(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA Error in " << __FILE__ << " at line "       \
                      << __LINE__ << " : " << cudaGetErrorString(err)      \
                      << std::endl;                                        \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

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
	out_rc_host_mirror.d_primitive_references = nullptr;
	out_rc_host_mirror.d_morton_codes = nullptr;
	out_rc_host_mirror.d_all_suns = nullptr; 
	out_rc_host_mirror.d_all_bulbs = nullptr; 
	out_rc_host_mirror.d_all_planes = nullptr;
	out_rc_host_mirror.d_lbvh_nodes = nullptr;
}

void copyConfigDataToDevice(const StlConfig& host_stl, RawConfig& rc_with_device_ptrs) {
	// Create temporary host-side SoA structs to hold the component device pointers
	SphereDataSoA   host_spheres_soa_temp;
	TriangleDataSoA host_triangles_soa_temp;

	// --- Copy SoA Primitive Data for Spheres (Component Arrays) ---
	if (rc_with_device_ptrs.num_spheres > 0) {
		// Host-side temporary vectors for de-interleaving AoS data
		std::vector<point3>    host_sphere_c(rc_with_device_ptrs.num_spheres);
		std::vector<float>     host_sphere_r(rc_with_device_ptrs.num_spheres);
		std::vector<Materials> host_sphere_mat(rc_with_device_ptrs.num_spheres);

		// De-interleave data
		for (int i = 0; i < rc_with_device_ptrs.num_spheres; ++i) {
			host_sphere_c[i]   = host_stl.host_spheres_data[i].c;
			host_sphere_r[i]   = host_stl.host_spheres_data[i].r;
			host_sphere_mat[i] = host_stl.host_spheres_data[i].mat;
		}

		// Allocate device memory for each component array and store pointers in TEMP host struct
		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&host_spheres_soa_temp.c,   rc_with_device_ptrs.num_spheres * sizeof(point3)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(host_spheres_soa_temp.c, host_sphere_c.data(), rc_with_device_ptrs.num_spheres * sizeof(point3), cudaMemcpyHostToDevice));
		
		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&host_spheres_soa_temp.r,   rc_with_device_ptrs.num_spheres * sizeof(float)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(host_spheres_soa_temp.r, host_sphere_r.data(), rc_with_device_ptrs.num_spheres * sizeof(float), cudaMemcpyHostToDevice));

		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&host_spheres_soa_temp.mat, rc_with_device_ptrs.num_spheres * sizeof(Materials)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(host_spheres_soa_temp.mat, host_sphere_mat.data(), rc_with_device_ptrs.num_spheres * sizeof(Materials), cudaMemcpyHostToDevice));

		// Allocate space for the SphereDataSoA struct itself on the device
		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&rc_with_device_ptrs.d_spheres_soa, sizeof(SphereDataSoA))); // Use the pointer from RawConfig
		// Copy the temporary host struct (containing device pointers .c, .r, .mat) to the device struct location
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(rc_with_device_ptrs.d_spheres_soa,
										   &host_spheres_soa_temp, // Source: address of TEMP host struct
										   sizeof(SphereDataSoA), 
										   cudaMemcpyHostToDevice));

	} else {
		rc_with_device_ptrs.d_spheres_soa = nullptr; // Ensure device pointer is null
	}

	// --- Copy SoA Primitive Data for Triangles (Component Arrays) ---
	if (rc_with_device_ptrs.num_triangles > 0) {
		// Host-side temporary vectors
		std::vector<point3>  host_triangle_p0(rc_with_device_ptrs.num_triangles);
		std::vector<point3>  host_triangle_p1(rc_with_device_ptrs.num_triangles);
		std::vector<point3>  host_triangle_p2(rc_with_device_ptrs.num_triangles);
		std::vector<vec3>    host_triangle_nor(rc_with_device_ptrs.num_triangles);
		std::vector<point3>  host_triangle_e1(rc_with_device_ptrs.num_triangles);
		std::vector<point3>  host_triangle_e2(rc_with_device_ptrs.num_triangles);
		std::vector<Materials> host_triangle_mat(rc_with_device_ptrs.num_triangles);

		// De-interleave data
		for (int i = 0; i < rc_with_device_ptrs.num_triangles; ++i) {
			host_triangle_p0[i]  = host_stl.host_triangles_data[i].p0;
			host_triangle_p1[i]  = host_stl.host_triangles_data[i].p1;
			host_triangle_p2[i]  = host_stl.host_triangles_data[i].p2;
			host_triangle_nor[i] = host_stl.host_triangles_data[i].nor;
			host_triangle_e1[i]  = host_stl.host_triangles_data[i].e1;
			host_triangle_e2[i]  = host_stl.host_triangles_data[i].e2;
			host_triangle_mat[i] = host_stl.host_triangles_data[i].mat;
		}

		// Allocate device memory for each component array and store pointers in TEMP host struct
		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&host_triangles_soa_temp.p0, rc_with_device_ptrs.num_triangles * sizeof(point3)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(host_triangles_soa_temp.p0, host_triangle_p0.data(), rc_with_device_ptrs.num_triangles * sizeof(point3), cudaMemcpyHostToDevice));
		
		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&host_triangles_soa_temp.p1, rc_with_device_ptrs.num_triangles * sizeof(point3)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(host_triangles_soa_temp.p1, host_triangle_p1.data(), rc_with_device_ptrs.num_triangles * sizeof(point3), cudaMemcpyHostToDevice));

		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&host_triangles_soa_temp.p2, rc_with_device_ptrs.num_triangles * sizeof(point3)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(host_triangles_soa_temp.p2, host_triangle_p2.data(), rc_with_device_ptrs.num_triangles * sizeof(point3), cudaMemcpyHostToDevice));

		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&host_triangles_soa_temp.nor, rc_with_device_ptrs.num_triangles * sizeof(vec3)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(host_triangles_soa_temp.nor, host_triangle_nor.data(), rc_with_device_ptrs.num_triangles * sizeof(vec3), cudaMemcpyHostToDevice));

		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&host_triangles_soa_temp.e1, rc_with_device_ptrs.num_triangles * sizeof(point3)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(host_triangles_soa_temp.e1, host_triangle_e1.data(), rc_with_device_ptrs.num_triangles * sizeof(point3), cudaMemcpyHostToDevice));

		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&host_triangles_soa_temp.e2, rc_with_device_ptrs.num_triangles * sizeof(point3)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(host_triangles_soa_temp.e2, host_triangle_e2.data(), rc_with_device_ptrs.num_triangles * sizeof(point3), cudaMemcpyHostToDevice));
		
		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&host_triangles_soa_temp.mat, rc_with_device_ptrs.num_triangles * sizeof(Materials)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(host_triangles_soa_temp.mat, host_triangle_mat.data(), rc_with_device_ptrs.num_triangles * sizeof(Materials), cudaMemcpyHostToDevice));

		// Allocate space for the TriangleDataSoA struct itself on the device
		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&rc_with_device_ptrs.d_triangles_soa, sizeof(TriangleDataSoA))); // Use the pointer from RawConfig
		// Copy the temporary host struct (containing device pointers .p0, .p1, etc.) to the device struct location
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(rc_with_device_ptrs.d_triangles_soa,
										   &host_triangles_soa_temp, // Source: address of TEMP host struct
										   sizeof(TriangleDataSoA),
										   cudaMemcpyHostToDevice));
	} else {
		rc_with_device_ptrs.d_triangles_soa = nullptr; // Ensure device pointer is null
	}

	// --- Primitive References and Morton Codes ---
	if (rc_with_device_ptrs.num_total_primitives > 0) {
		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&rc_with_device_ptrs.d_primitive_references, rc_with_device_ptrs.num_total_primitives * sizeof(PrimitiveReference)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(rc_with_device_ptrs.d_primitive_references, host_stl.host_primitive_references.data(), rc_with_device_ptrs.num_total_primitives * sizeof(PrimitiveReference), cudaMemcpyHostToDevice));

		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&rc_with_device_ptrs.d_morton_codes, rc_with_device_ptrs.num_total_primitives * sizeof(unsigned int)));
	} else {
		rc_with_device_ptrs.d_primitive_references = nullptr;
		rc_with_device_ptrs.d_morton_codes = nullptr;
	}

	// --- Lights, Planes, etc. ---
	// (Allocation logic as before, ensuring pointers are null if count is zero)
	if (rc_with_device_ptrs.num_sun > 0) {
		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&rc_with_device_ptrs.d_all_suns, rc_with_device_ptrs.num_sun * sizeof(Sun)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(rc_with_device_ptrs.d_all_suns, host_stl.sun_data.data(), rc_with_device_ptrs.num_sun * sizeof(Sun), cudaMemcpyHostToDevice));
	} else {
		rc_with_device_ptrs.d_all_suns = nullptr;
	}
	if (rc_with_device_ptrs.num_bulbs > 0) {
		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&rc_with_device_ptrs.d_all_bulbs, rc_with_device_ptrs.num_bulbs * sizeof(Bulb)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(rc_with_device_ptrs.d_all_bulbs, host_stl.bulb_data.data(), rc_with_device_ptrs.num_bulbs * sizeof(Bulb), cudaMemcpyHostToDevice));
	} else {
		rc_with_device_ptrs.d_all_bulbs = nullptr;
	}
	if (rc_with_device_ptrs.num_planes > 0) {
		CUDA_CHECK_CONFIG_UTILS(cudaMalloc(&rc_with_device_ptrs.d_all_planes, rc_with_device_ptrs.num_planes * sizeof(Plane)));
		CUDA_CHECK_CONFIG_UTILS(cudaMemcpy(rc_with_device_ptrs.d_all_planes, host_stl.plane_data.data(), rc_with_device_ptrs.num_planes * sizeof(Plane), cudaMemcpyHostToDevice));
	} else {
		rc_with_device_ptrs.d_all_planes = nullptr;
	}
}

// Frees all device memory pointed to by members of rc_with_device_ptrs.
void freeRawConfigDeviceMemory(RawConfig& rc_with_device_ptrs) {
	// Free the device copies of the SoA structs themselves
	// It's safe to call cudaFree on nullptr
	cudaFree(rc_with_device_ptrs.d_spheres_soa); 
	rc_with_device_ptrs.d_spheres_soa = nullptr; 
	
	cudaFree(rc_with_device_ptrs.d_triangles_soa); 
	rc_with_device_ptrs.d_triangles_soa = nullptr; 

	// Free Primitive References and Morton codes
	cudaFree(rc_with_device_ptrs.d_primitive_references); 
	rc_with_device_ptrs.d_primitive_references = nullptr;
	
	cudaFree(rc_with_device_ptrs.d_morton_codes); 
	rc_with_device_ptrs.d_morton_codes = nullptr;

	// Free Lights, Planes, etc.
	cudaFree(rc_with_device_ptrs.d_all_suns); 
	rc_with_device_ptrs.d_all_suns = nullptr;
	
	cudaFree(rc_with_device_ptrs.d_all_bulbs); 
	rc_with_device_ptrs.d_all_bulbs = nullptr;

	cudaFree(rc_with_device_ptrs.d_all_planes); 
	rc_with_device_ptrs.d_all_planes = nullptr;

	// Free LBVH nodes
	cudaFree(rc_with_device_ptrs.d_lbvh_nodes); 
	rc_with_device_ptrs.d_lbvh_nodes = nullptr;
}

