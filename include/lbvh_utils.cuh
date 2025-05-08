#ifndef LBVH_UTILS_CUH
#define LBVH_UTILS_CUH

#include "config.hpp" // For RawConfig, AABB
#include "object.cuh" // For PrimitiveReference, Sphere, Triangle

// Kernel to generate Morton codes
__global__ void generate_morton_codes_kernel(
	const PrimitiveReference* d_primitive_refs, // Input: array of primitive references
	const Sphere* d_all_spheres_data,           // Input: SoA sphere data
	const Triangle* d_all_triangles_data,       // Input: SoA triangle data
	unsigned int* d_out_morton_codes,           // Output: Morton codes
	int num_total_primitives,
	const AABB& scene_bounds,
	int morton_bits_per_dim
);

// Host function to orchestrate Morton code generation and sorting
// Takes RawConfig by reference, which already contains allocated device pointers
// for primitive data. It will fill d_morton_codes and sort d_primitive_references.
void build_morton_codes_and_sort_primitives(RawConfig& config_with_device_data);

#endif // LBVH_UTILS_CUH