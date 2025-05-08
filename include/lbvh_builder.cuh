#ifndef LBVH_BUILDER_CUH
#define LBVH_BUILDER_CUH

#include "config.hpp" // For RawConfig, which holds Morton codes, primitive refs, and will hold BVH nodes
#include "object.cuh" // For AABB, PrimitiveReference, Sphere, Triangle
#include "lbvh.cuh"   // For LBVHNode

// Main host function to orchestrate the LBVH construction using Karas' algorithm
// N is the number of primitives.
// BVH will have N leaf nodes and N-1 internal nodes. Total 2N-1 nodes.
// Leaf nodes can be indexed from N-1 to 2N-2.
// Internal nodes can be indexed from 0 to N-2. Root is node 0.
// morton_bits typically 30 (10 bits per dimension).
void build_lbvh_karas(RawConfig& config, int morton_bits = 30);

#endif // LBVH_BUILDER_CUH