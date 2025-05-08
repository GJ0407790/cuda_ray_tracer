#include "lbvh_utils.cuh"
#include "vec3.cuh"         // For point3, vec3 operations

#include <thrust/sort.h>
#include <thrust/device_ptr.h> // For thrust::device_ptr
#include <thrust/execution_policy.h> // For thrust::device
#include <stdio.h> // For printf in kernel if debugging

// Morton code helper functions (same as previously discussed)
__device__ unsigned int expand_bits(unsigned int v) 
{
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__device__ unsigned int morton_3d(unsigned int x, unsigned int y, unsigned int z) 
{
  return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2);
}

__device__ unsigned int quantize_coordinate(float coord, float scene_min_coord, float scene_range, int bits) 
{
  if (scene_range <= 1e-6f) return 0; // Avoid division by zero or if range is negligible
  float normalized = (coord - scene_min_coord) / scene_range;
  normalized = fmaxf(0.0f, fminf(1.0f, normalized)); // Clamp to [0, 1]
  return static_cast<unsigned int>(normalized * ((1 << bits) - 1));
}

__global__ void generate_morton_codes_kernel(
  const PrimitiveReference* d_primitive_refs,
  const SphereDataSoA& d_spheres_soa,
  const TriangleDataSoA& d_triangles_soa,
  unsigned int* d_out_morton_codes,
  int num_total_primitives,
  float3 scene_min_corner,
  float3 scene_max_corner,
  int morton_bits_per_dim)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_total_primitives) return;

  const PrimitiveReference& ref = d_primitive_refs[idx];
  point3 centroid;

  if (ref.type == PrimitiveType::SPHERE) 
  {
    centroid = d_spheres_soa.c[ref.id_in_type_array];
  } 
  else if (ref.type == PrimitiveType::TRIANGLE) 
  {
    // Access vertices from SoA
    point3 p0 = d_triangles_soa.p0[ref.id_in_type_array];
    point3 p1 = d_triangles_soa.p1[ref.id_in_type_array];
    point3 p2 = d_triangles_soa.p2[ref.id_in_type_array];
    centroid = (p0 + p1 + p2) / 3.0f;
  } 
  else 
  {
    d_out_morton_codes[idx] = 0xFFFFFFFF; // Invalid type or error
    return;
  }

  float scene_range_x = scene_max_corner.x - scene_min_corner.x;
  float scene_range_y = scene_max_corner.y - scene_min_corner.y;
  float scene_range_z = scene_max_corner.z - scene_min_corner.z;

  unsigned int qx = quantize_coordinate(centroid.x, scene_min_corner.x, scene_range_x, morton_bits_per_dim);
  unsigned int qy = quantize_coordinate(centroid.y, scene_min_corner.y, scene_range_y, morton_bits_per_dim);
  unsigned int qz = quantize_coordinate(centroid.z, scene_min_corner.z, scene_range_z, morton_bits_per_dim);

  d_out_morton_codes[idx] = morton_3d(qx, qy, qz);
}

void build_morton_codes_and_sort_primitives(RawConfig& config_with_device_data) {
  if (config_with_device_data.num_total_primitives == 0) 
  {
    printf("No primitives to process for Morton codes.\n");
    return;
  }

  constexpr int morton_bits = 10; // For 30-bit Morton codes (10 bits per dimension)
  constexpr int block_size = 256;
  int grid_size = (config_with_device_data.num_total_primitives - 1) / block_size + 1;

  // Generate Morton codes
  generate_morton_codes_kernel<<<grid_size, block_size>>>(
    config_with_device_data.d_primitive_references, // Input refs (unsorted at this point)
    *config_with_device_data.d_spheres_soa,
    *config_with_device_data.d_triangles_soa,
    config_with_device_data.d_morton_codes,         // Output Morton codes
    config_with_device_data.num_total_primitives,
    config_with_device_data.scene_min_corner,
    config_with_device_data.scene_max_corner,
    morton_bits
  );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Morton generation kernel failed: %s\n", cudaGetErrorString(err));
    return;
  }
  // cudaDeviceSynchronize(); // Optional: for debugging

  // Sort d_primitive_references based on d_morton_codes
  // d_morton_codes will also be sorted as keys.
  try {
    thrust::sort_by_key(
      thrust::device, // Execution policy
      thrust::device_ptr<unsigned int>(config_with_device_data.d_morton_codes), // Keys: Morton codes
      thrust::device_ptr<unsigned int>(config_with_device_data.d_morton_codes + config_with_device_data.num_total_primitives),
      thrust::device_ptr<PrimitiveReference>(config_with_device_data.d_primitive_references) // Values: Primitive references
    );
  } catch (const thrust::system_error& e) {
    fprintf(stderr, "Thrust sort_by_key failed: %s\n", e.what());
    cudaError_t sort_err = cudaGetLastError();
    if (sort_err != cudaSuccess) {
      fprintf(stderr, "CUDA error after Thrust exception: %s\n", cudaGetErrorString(sort_err));
    }
    return;
  }
  // cudaDeviceSynchronize(); // Optional: for debugging

  // printf("Morton codes generated and primitive references sorted successfully.\n");
  // Now, config_with_device_data.d_primitive_references is sorted by Morton code.
  // config_with_device_data.d_morton_codes is also sorted.
}