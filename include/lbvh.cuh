#ifndef LBVH_CUH
#define LBVH_CUH

#include "object.cuh"

struct LBVHNode {
  AABB bbox;

  // For internal nodes
  unsigned int left_child_offset;
  unsigned int right_child_offset;

  // For leaf node
  unsigned int primitive_offset;

  // For internal nodes: 0.
  // For leaf nodes: number of primitives in this leaf.
  // A common trick: store (num_primitives << 2) | type_flags. Here, a simple count.
  // If num_primitives > 0, it's a leaf. Max primitives per leaf can be small (e.g., 1-4).
  unsigned short num_primitives_in_leaf;

  // For bottom up internal node's AABB construction
  unsigned int visited_atomic_counter = 0;

  __host__ __device__ bool isLeaf() const {
    return num_primitives_in_leaf > 0;
  }
};

#endif // LBVH_CUH
