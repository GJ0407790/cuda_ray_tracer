#ifndef BVH_TRAVERSAL_CUH
#define BVH_TRAVERSAL_CUH

#include "object.cuh" // For Ray, ObjectInfo, LBVHNode
#include "config.hpp" // For RawConfig

// Traverses the LBVH to find the closest intersection for the given ray.
// Modifies ray.t_max if a closer hit is found.
// Returns ObjectInfo for the closest hit.
__device__ ObjectInfo traverse_lbvh(
  const Ray& ray,
  const RawConfig* config, // Provides access to BVH nodes and scene geometry
  float initial_t_max     // Initial maximum distance for the ray
);

#endif // BVH_TRAVERSAL_CUH