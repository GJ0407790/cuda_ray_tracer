#include "lbvh_builder.cuh"
#include "lbvh_utils.cuh" // For morton code utilities (though sorting is done before)
#include "vec3.cuh"
#include "interval.cuh"     // For AABB definition
#include "object.cuh"       // For PrimitiveReference, Sphere, Triangle, AABB
#include "lbvh.cuh"         // For LBVHNode

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda_runtime.h>
#include <stdio.h> // For printf debugging

// CUDA error checking macro
#define CUDA_CHECK(err)                                     \
    do {                                                    \
        cudaError_t err_ = (err);                           \
        if (err_ != cudaSuccess) {                          \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err_)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)


// Helper device function to get AABB of a primitive
__device__ AABB get_primitive_aabb_device( // Renamed to avoid conflict if there's a host version
    const PrimitiveReference& ref,
    const SphereDataSoA& spheres_soa,
    const TriangleDataSoA& triangles_soa)
{
  if (ref.type == PrimitiveType::SPHERE) 
  {
    point3 center = spheres_soa.c[ref.id_in_type_array];
    float radius = spheres_soa.r[ref.id_in_type_array];
    vec3 radius_vec(radius, radius, radius);

    return AABB(center - radius_vec, center + radius_vec);
  } 
  else if (ref.type == PrimitiveType::TRIANGLE) 
  {
    point3 p0 = triangles_soa.p0[ref.id_in_type_array];
    point3 p1 = triangles_soa.p1[ref.id_in_type_array];
    point3 p2 = triangles_soa.p2[ref.id_in_type_array];

    return AABB(p0, p1, p2);
  }

  // Return an empty/invalid AABB if type is unknown or ref is invalid
  return AABB();
}

__global__ void initialize_leaf_nodes_kernel(
  LBVHNode* d_bvh_nodes,
  unsigned int num_primitives,
  unsigned int leaf_node_start_idx)
{
  unsigned int primitive_sorted_idx = blockIdx.x * blockDim.x + threadIdx.x; // Corresponds to 0 to N-1
  if (primitive_sorted_idx >= num_primitives) return;

  unsigned int current_leaf_global_idx = leaf_node_start_idx + primitive_sorted_idx;

  d_bvh_nodes[current_leaf_global_idx].num_primitives_in_leaf = 1;
  d_bvh_nodes[current_leaf_global_idx].primitive_offset = primitive_sorted_idx;
}

// Note: The reference 'delta' uses indices 'a' and 'b' for tie-breaking.
// These are indices into the sorted_morton_codes array.
// 'ka' is 'codes[a]'. 'n' is num_primitives.
__device__ __forceinline__ int adapted_delta(
  int idx_a, int idx_b, // Indices of the primitives in the sorted list
  unsigned int num_primitives,
  const unsigned int *sorted_morton_codes)
{
  // Ensure indices are valid for accessing sorted_morton_codes
  // The reference checks b < 0 || b > n - 1. Here, both indices should be checked.
  // The original primitives considered are at sorted_morton_codes[idx_a] and sorted_morton_codes[idx_b]
  bool invalid_a = (idx_a < 0 || idx_a >= num_primitives);
  bool invalid_b = (idx_b < 0 || idx_b >= num_primitives);

  if (invalid_a || invalid_b) return -1; // Or handle error appropriately

  unsigned int ka = sorted_morton_codes[idx_a];
  unsigned int kb = sorted_morton_codes[idx_b];

  if (ka == kb) 
  {
    // If keys are equal, use original indices (which are 'idx_a' and 'idx_b' themselves
    // if they represent distinct positions in the sorted list) as a fallback.
    // Add 32 (or morton_bits + 1) to ensure this delta is larger than any differing-code delta.
    return 32 + __clz((unsigned int) idx_a ^ (unsigned int) idx_b);
  }
  // clz = count leading zeros
  return __clz(ka ^ kb);
}

__device__ __forceinline__ int2 determine_range_adapted(
	const unsigned int *sorted_morton_codes,
	unsigned int n, // num_primitives
	int i)          // current internal node index (0 to n-2)
{
	// Calculate deltas relative to split point 'i'
	// delta_l compares primitive i with primitive i-1
	// delta_r compares primitive i with primitive i+1
	const int delta_l = adapted_delta(i, i - 1, n, sorted_morton_codes);
	const int delta_r = adapted_delta(i, i + 1, n, sorted_morton_codes);

	// Determine direction 'd' and minimum delta 'delta_min_val'
	// The direction points towards the neighbor with the LARGER delta (longer shared prefix).
	// The minimum delta determines how far we need to search.
	int d; // direction (+1 or -1)
	int delta_min_val;

	// Handle edge case N=1 (no internal nodes, kernel shouldn't run, but safe check)
	if (n <= 1) {
	    return make_int2(i, i);
	}

	// Determine search direction and threshold delta
	if (delta_r > delta_l) { // Search RIGHT (positive direction)
		d = 1;
		delta_min_val = delta_l; // Threshold is the delta to the non-search side (or -1 if invalid)
	} else { // Search LEFT (negative direction)
		d = -1;
		delta_min_val = delta_r; // Threshold is the delta to the non-search side (or -1 if invalid)
	}

    // Need to handle case where one neighbor is invalid (delta = -1)
    // If delta_l == -1, delta_min_val becomes delta_r, d becomes 1 (search right).
    // If delta_r == -1, delta_min_val becomes delta_l, d becomes -1 (search left).
    // If both are -1 (only possible if N=1, handled above), delta_min_val = -1.


	// Compute upper bound 'l_max' for the search length using exponential search
	unsigned int l_max = 1; // Start checking offset 1*d, then 2*d, 4*d, etc.
	int neighbor_idx_exp = i + l_max * d;

	// Check initial neighbor before loop
	int current_delta_exp = adapted_delta(i, neighbor_idx_exp, n, sorted_morton_codes);

	while (current_delta_exp > delta_min_val) {
		l_max <<= 1; // Double search distance
		neighbor_idx_exp = i + l_max * d;
		// Check bounds before calling delta
		if (neighbor_idx_exp < 0 || neighbor_idx_exp >= (int)n) {
			break; // Stop if index goes out of bounds
		}
		current_delta_exp = adapted_delta(i, neighbor_idx_exp, n, sorted_morton_codes);
	}
	// l_max is now >= the power of 2 that bounds the search range length

	// Binary search for the precise length 'l' within the range [0, l_max-1]
	// Find largest 'l' such that delta(i, i + l*d) > delta_min_val
	unsigned int l = 0;

  for (unsigned int t = l_max >> 1; t > 0; t >>= 1) {
		int neighbor_idx_bin = i + (l + t) * d;

		// Check bounds before calling delta
		if (neighbor_idx_bin >= 0 && neighbor_idx_bin < (int)n) {
			int current_delta_bin = adapted_delta(i, neighbor_idx_bin, n, sorted_morton_codes);

			if (current_delta_bin > delta_min_val) {
				l += t; // Extend the range, accept this step
			}
		}
	}
	// 'l' is now the largest offset from 'i' in direction 'd' that satisfies the delta condition

	const int j = i + l * d; // Calculate the index of the other end of the range


	// The range of primitives associated with the split at 'i' is [min(i,j), max(i,j)]
	int2 result_range = (i < j) ? make_int2(i, j) : make_int2(j, i);
	return result_range;
}


// first and last are indices of primitives in the sorted list
__device__ __forceinline__ int find_split_adapted(
  const unsigned int *sorted_morton_codes,
  int first, // first primitive index in the range
  int last,  // last primitive index in the range
  unsigned int n) // total number of primitives
{
  if (first == last) return first; // Range of one, split is itself

  // Calculate the number of highest bits that are the same for the entire range [first, last]
  const int common_prefix = adapted_delta(first, last, n, sorted_morton_codes);

  // Use binary search to find where the next bit differs.
  // We are looking for the highest object (split) in [first, last-1]
  // that shares more than commonPrefix bits with the 'first' one.
  // The split occurs *after* this 'split' primitive.
  int split_primitive_idx = first; // initial guess: last primitive in the left child's group
  int step = last - first;

  do {
    step = (step + 1) >> 1; // exponential decrease
    const int new_potential_split = split_primitive_idx + step;

    if (new_potential_split < last) 
    { // new_potential_split is a candidate for the end of left group
      // How many bits does 'first' share with 'new_potential_split'?
      const int split_prefix = adapted_delta(first, new_potential_split, n, sorted_morton_codes);
      
      if (split_prefix > common_prefix) 
      {
        split_primitive_idx = new_potential_split; // Accept: [first...new_potential_split] shares more.
      }
    }
  } while (step > 1);

  return split_primitive_idx; // Index of the last primitive in the left part of the split
}

// --- Main Hierarchy Construction Kernel (Using Refined Child Logic) ---
__global__ void generate_internal_nodes_karas_kernel(
  const int num_lbvh_nodes,
  LBVHNode* d_bvh_nodes,
  int* d_parent_indices,
  const unsigned int* d_sorted_morton_codes,
  unsigned int num_primitives,
  int morton_bits)
{
  unsigned int internal_node_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (internal_node_idx >= num_primitives - 1) return;

  const int2 prim_range = determine_range_adapted(d_sorted_morton_codes, num_primitives, internal_node_idx);

  // Ensure range is valid before proceeding
  if (prim_range.x > prim_range.y) {
       return; // Avoid further execution with bad range
  }

  const int split_prim_idx = find_split_adapted(d_sorted_morton_codes, prim_range.x, prim_range.y, num_primitives);

   // Ensure split is valid before proceeding
  if (split_prim_idx < prim_range.x || split_prim_idx >= prim_range.y) {
      // split_prim_idx must be within [prim_range.x, prim_range.y - 1]
       // Handle error? For now, maybe default children to invalid indices or return.
       // Setting children to an invalid index might be better for debugging later stages.
       d_bvh_nodes[internal_node_idx].left_child_offset = 0xFFFFFFFF;
       d_bvh_nodes[internal_node_idx].right_child_offset = 0xFFFFFFFF;
       d_bvh_nodes[internal_node_idx].num_primitives_in_leaf = 0; // Still mark as internal maybe
       return;
  }

  // --- Determine children global indices using Reference's Delta Comparison Logic ---
  unsigned int leaf_node_base_idx = num_primitives - 1;
  unsigned int left_child_global_idx;
  unsigned int right_child_global_idx;

  // Calculate delta at the optimal split point 'split_prim_idx' for the range.
  // This delta is between primitive 'split_prim_idx' and 'split_prim_idx + 1'.
  int delta_at_split = adapted_delta(split_prim_idx, split_prim_idx + 1, num_primitives, d_sorted_morton_codes);

  // --- Determine Left Child ---
  if (split_prim_idx == prim_range.x) {
      left_child_global_idx = leaf_node_base_idx + split_prim_idx; // Range starts == split -> Leaf
  } else {
      int delta_left_check = adapted_delta(prim_range.x, split_prim_idx, num_primitives, d_sorted_morton_codes);
      if (delta_left_check > delta_at_split) {
          left_child_global_idx = split_prim_idx; // Child is Internal Node 'split_prim_idx'
      } else {
          left_child_global_idx = leaf_node_base_idx + prim_range.x; // Child is Leaf Node 'prim_range.x'
      }
  }

  // --- Determine Right Child ---
  if (split_prim_idx + 1 == prim_range.y) {
      right_child_global_idx = leaf_node_base_idx + prim_range.y; // Range end is split+1 -> Leaf
  } else {
      int delta_right_check = adapted_delta(split_prim_idx + 1, prim_range.y, num_primitives, d_sorted_morton_codes);
       if (delta_right_check > delta_at_split) {
           right_child_global_idx = split_prim_idx + 1; // Child is Internal Node 'split_prim_idx + 1'
       } else {
           right_child_global_idx = leaf_node_base_idx + prim_range.y; // Child is Leaf Node 'prim_range.y'
       }
  }

  // --- Store children in d_bvh_nodes[internal_node_idx] ---
  if (internal_node_idx < num_primitives - 1) { // Re-check bounds just in case
      LBVHNode& current_internal_node = d_bvh_nodes[internal_node_idx];
      current_internal_node.num_primitives_in_leaf = 0;
      current_internal_node.left_child_offset = left_child_global_idx;
      current_internal_node.right_child_offset = right_child_global_idx;
  } else {
      return;
  }

  // --- Populate the parent indices array ---
  if (d_parent_indices != nullptr) {
       // Bounds check before writing parent indices
      bool parent_write_ok = true;
      if (left_child_global_idx >= num_lbvh_nodes) {
           parent_write_ok = false;
      }
       if (right_child_global_idx >= num_lbvh_nodes) {
           parent_write_ok = false;
       }

      if (parent_write_ok) {
          // Use atomic operations if there's any chance of race conditions,
          // but theoretically each child should only have one parent assigned.
          // Standard write should be okay if logic is correct.
          d_parent_indices[left_child_global_idx] = internal_node_idx;
          d_parent_indices[right_child_global_idx] = internal_node_idx;
      }

      // Mark root's parent
      if (internal_node_idx == 0) {
          d_parent_indices[internal_node_idx] = -1;
      }
  }
}

__global__ void set_aabb_kernel_adapted(
  LBVHNode* d_bvh_nodes,
  const int* d_parent_indices,
  const PrimitiveReference* d_primitive_refs, // From RawConfig.d_primitive_references (sorted)
  const SphereDataSoA& spheres_soa,           
  const TriangleDataSoA& triangles_soa,       
  unsigned int num_primitives,                // N
  unsigned int leaf_node_start_idx)
{
  unsigned int primitive_sorted_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (primitive_sorted_idx >= num_primitives) return;

  // 1. Current node is the leaf node this thread is responsible for.
  unsigned int current_node_global_idx = leaf_node_start_idx + primitive_sorted_idx;
  LBVHNode& leaf_node = d_bvh_nodes[current_node_global_idx];

  // 2. Calculate and set AABB for this leaf node.
  //    leaf_node.primitive_offset should already be set by initialize_leaf_nodes_kernel
  //    to primitive_sorted_idx.
  leaf_node.bbox = get_primitive_aabb_device(
      d_primitive_refs[leaf_node.primitive_offset], // Use the offset stored in the leaf
      spheres_soa,
      triangles_soa);

  // 3. Traverse upwards to update parent AABBs.
  int parent_global_idx = d_parent_indices[current_node_global_idx];

  // Loop while current_node_global_idx has a valid parent.
  // The root's parent is typically -1 or some other invalid index.
  while (parent_global_idx != -1 && parent_global_idx < (num_primitives -1) ) // Parent must be an internal node (idx < N-1)
  {
    LBVHNode& parent_node = d_bvh_nodes[parent_global_idx];

    // Atomically increment the visited counter of the parent.
    // This counter should have been initialized to 0 for all internal nodes.
    unsigned int previous_visited_count = atomicAdd(&parent_node.visited_atomic_counter, 1);

    // If previous_visited_count is 0, this thread is the first child to arrive.
    // Its sibling's AABB is not yet guaranteed to be ready. So, this path terminates.
    if (previous_visited_count == 0) 
    {
      break;
    }

    // If previous_visited_count is 1, this thread is the second child to arrive.
    // Both children of parent_node have had their AABBs computed.
    // Now, compute parent_node's AABB.
    // Children indices are stored in parent_node.left_child_offset and parent_node.right_child_offset
    unsigned int left_child_idx = parent_node.left_child_offset;
    unsigned int right_child_idx = parent_node.right_child_offset;

    // Ensure child indices are valid before dereferencing (important for robustness)
    // This check might be excessive if hierarchy construction guarantees validity.
    // if (left_child_idx < config.num_lbvh_nodes && right_child_idx < config.num_lbvh_nodes) {
    const AABB& left_child_bbox = d_bvh_nodes[left_child_idx].bbox;
    const AABB& right_child_bbox = d_bvh_nodes[right_child_idx].bbox;

    parent_node.bbox = AABB(left_child_bbox, right_child_bbox);

    // Move up to the next parent.
    current_node_global_idx = parent_global_idx; // The parent becomes the current node for the next iteration
    parent_global_idx = d_parent_indices[current_node_global_idx];
  }
}

__global__ void initialize_visited_counters_kernel(LBVHNode* d_bvh_nodes, int num_internal_nodes) {
  unsigned int internal_node_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (internal_node_idx < num_internal_nodes) 
  {
    // Internal nodes are indexed 0 to num_internal_nodes-1
    d_bvh_nodes[internal_node_idx].visited_atomic_counter = 0;
  }
}


// Main host function to orchestrate the LBVH construction using Karas' algorithm
void build_lbvh_karas(RawConfig& config, int morton_bits /* = 30 */) 
{
  cudaEvent_t start_event, stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));
  CUDA_CHECK(cudaEventRecord(start_event));

  int N = config.num_total_primitives;
  unsigned int num_internal_nodes = N - 1;
  unsigned int num_leaf_nodes = N;
  config.num_lbvh_nodes = num_leaf_nodes + num_internal_nodes;

  CUDA_CHECK(cudaMalloc(&config.d_lbvh_nodes, config.num_lbvh_nodes * sizeof(LBVHNode)));

  // Temporary array for parent indices
  // Size: config.num_lbvh_nodes, stores the parent index for each node. Root has parent -1 or UINT_MAX.
  int* d_parent_indices = nullptr;
  CUDA_CHECK(cudaMalloc(&d_parent_indices, config.num_lbvh_nodes * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_parent_indices, 0xFF, config.num_lbvh_nodes * sizeof(int)));

  // --- Stage 0: Calculate Morton code and sort primitive accordingly ---
  build_morton_codes_and_sort_primitives(config);

  constexpr int threads_per_block = 256;
  // --- Stage 1: Initialize Leaf Nodes ---
  // Leaf nodes are stored at indices [num_internal_nodes, config.num_lbvh_nodes - 1]
  // Or, if internal nodes are [0, N-2] and leaves [N-1, 2N-2]:
  unsigned int leaf_node_start_idx = num_internal_nodes; 

  int grid_dim_leaf_init = (num_leaf_nodes - 1) / threads_per_block + 1;
  initialize_leaf_nodes_kernel<<<grid_dim_leaf_init, threads_per_block>>>(
      config.d_lbvh_nodes,
      num_leaf_nodes, // N
      leaf_node_start_idx
  );

  CUDA_CHECK(cudaGetLastError());

  // --- Stage 2: Generate Internal Node Hierarchy (Karas' algorithm) ---
  int grid_dim_internal_hierarchy = (num_internal_nodes - 1) / threads_per_block + 1;
  generate_internal_nodes_karas_kernel<<<grid_dim_internal_hierarchy, threads_per_block>>>(
    config.num_lbvh_nodes,
    config.d_lbvh_nodes,
    d_parent_indices,
    config.d_morton_codes, // Assumed to be sorted
    N,                     // num_primitives (num_leaf_nodes)
    morton_bits
  );
  CUDA_CHECK(cudaGetLastError());

  // --- Stage 3: Initialize Visited Counters for Internal Nodes ---
  int grid_dim_visited_init = (num_internal_nodes + threads_per_block - 1) / threads_per_block;
  initialize_visited_counters_kernel<<<grid_dim_visited_init, threads_per_block>>>(
    config.d_lbvh_nodes,
    num_internal_nodes // Pass the count of internal nodes
  );

  CUDA_CHECK(cudaGetLastError());

  // --- Stage 4: Calculate AABBs for all nodes (leaves then bottom-up for internal) ---
  // This kernel processes from leaves up
  int grid_dim_aabb = (num_leaf_nodes - 1) / threads_per_block + 1;
  set_aabb_kernel_adapted<<<grid_dim_aabb, threads_per_block>>>(
      config.d_lbvh_nodes,
      d_parent_indices,
      config.d_primitive_references, // Assumed to be sorted by Morton code
      *config.d_spheres_soa,
      *config.d_triangles_soa,
      N, // num_primitives (num_leaf_nodes)
      leaf_node_start_idx
  );
  CUDA_CHECK(cudaGetLastError());

  // Ensure all GPU work is done before measuring time and freeing memory
  CUDA_CHECK(cudaDeviceSynchronize());

  // Cleanup temporary arrays
  CUDA_CHECK(cudaFree(d_parent_indices));
  d_parent_indices = nullptr;

  // Set the bvh_root_node_idx or bvh_head in config if needed.
  // Root is node 0 if N > 1. If N=1, root is also node 0 (the only leaf).
  // This is implicit in how traversal would start with d_bvh_nodes[0].

  CUDA_CHECK(cudaEventRecord(stop_event));
  CUDA_CHECK(cudaEventSynchronize(stop_event));
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_event, stop_event));
  printf("LBVH Build time (N=%d): %.3f ms\n", N, milliseconds);

  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));

  printf("LBVH Build (Karas algorithm) complete. Total nodes: %u\n", config.num_lbvh_nodes);
  // Print leaf node info for debugging
  if (N <= 16) {
    LBVHNode* h_lbvh_nodes;
    CUDA_CHECK(cudaMallocHost(&h_lbvh_nodes, config.num_lbvh_nodes * sizeof(LBVHNode)));
    CUDA_CHECK(cudaMemcpy(h_lbvh_nodes, config.d_lbvh_nodes, config.num_lbvh_nodes * sizeof(LBVHNode), cudaMemcpyDeviceToHost));

    PrimitiveReference* h_primitive_references;
    CUDA_CHECK(cudaMallocHost(&h_primitive_references, config.num_total_primitives * sizeof(PrimitiveReference)));
    CUDA_CHECK(cudaMemcpy(h_primitive_references, config.d_primitive_references, config.num_total_primitives * sizeof(PrimitiveReference), cudaMemcpyDeviceToHost));

    printf("Node Info:\n");
    for (int i = 0; i < config.num_lbvh_nodes; ++i) {
      printf("  Node %d: num_primitives_in_leaf=%d, primitive_offset=%d, left_child_offset=%d, right_child_offset=%d, visited_atomic_counter=%u, bbox=(min: %.2f, %.2f, %.2f, max: %.2f, %.2f, %.2f)\n",
             i,
             h_lbvh_nodes[i].num_primitives_in_leaf,
             h_lbvh_nodes[i].primitive_offset,
             h_lbvh_nodes[i].left_child_offset,
             h_lbvh_nodes[i].right_child_offset,
             h_lbvh_nodes[i].visited_atomic_counter,
             h_lbvh_nodes[i].bbox.x.min, h_lbvh_nodes[i].bbox.y.min, h_lbvh_nodes[i].bbox.z.min,
             h_lbvh_nodes[i].bbox.x.max, h_lbvh_nodes[i].bbox.y.max, h_lbvh_nodes[i].bbox.z.max);
    }

    CUDA_CHECK(cudaFreeHost(h_lbvh_nodes));
    CUDA_CHECK(cudaFreeHost(h_primitive_references));
  }
}
