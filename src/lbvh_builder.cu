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
    const Sphere* d_all_spheres,
    const Triangle* d_all_triangles)
{
  if (ref.type == PrimitiveType::SPHERE) 
  {
    return d_all_spheres[ref.id_in_type_array].bbox;
  } 
  else if (ref.type == PrimitiveType::TRIANGLE) 
  {
    return d_all_triangles[ref.id_in_type_array].bbox;
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

// i is the index of the current primitive in the sorted list (0 to n-1)
__device__ __forceinline__ int2 determine_range_adapted(
  const unsigned int *sorted_morton_codes,
  unsigned int n, // num_primitives
  int i)          // current primitive index
{
  // Determine direction of the range (+1 or -1)
  // Compare with primitive i-1 and i+1
  const int delta_l = adapted_delta(i, i - 1, n, sorted_morton_codes);
  const int delta_r = adapted_delta(i, i + 1, n, sorted_morton_codes);

  int d; // direction
  int delta_min_val; // min of delta_r and delta_l
  
  if (delta_r < delta_l) 
  { // Note: original code logic here might be slightly different if -1 is returned by delta
    d = -1;
    delta_min_val = delta_r;
    if (delta_r == -1) delta_min_val = delta_l; // If right is invalid, use left
  } 
  else 
  {
    d = 1;
    delta_min_val = delta_l;
    if (delta_l == -1) delta_min_val = delta_r; // If left is invalid, use right
  }

  // If both are -1 (e.g., n=1, i=0), range is just i.
  if (delta_min_val == -1 && n == 1 && i == 0) return make_int2(0,0);

  // Compute upper bound of the length of the range
  unsigned int l_max = 2;
  // Ensure i + l_max * d is a valid index before calling delta
  while (adapted_delta(i, i + l_max * d, n, sorted_morton_codes) > delta_min_val) 
  {
    l_max <<= 1;
  }

  // Find other end using binary search
  unsigned int l = 0;
  for (unsigned int t = l_max >> 1; t > 0; t >>= 1) 
  {
    if (adapted_delta(i, i + (l + t) * d, n, sorted_morton_codes) > delta_min_val) 
    {
      l += t;
    }
  }
  const int j = i + l * d;

  // Ensure i <= j (or min_idx, max_idx)
  return i < j ? make_int2(i, j) : make_int2(j, i);
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

// --- Main Hierarchy Construction Kernel (Updated) ---
// Constructs internal nodes. Each thread 'internal_node_idx' (0 to N-2) builds d_bvh_nodes[internal_node_idx].
__global__ void generate_internal_nodes_karas_kernel(
  LBVHNode* d_bvh_nodes,                      // Combined array for internal and leaf nodes
  int* d_parent_indices,                      // Optional: parent_idx[node_global_idx] = parent_node_idx
  const unsigned int* d_sorted_morton_codes,  // Morton codes sorted
  unsigned int num_primitives,                // N = number of leaf/primitive nodes
  int morton_bits)                            // Effective bits in Morton code (passed to delta if needed, though current adapted_delta doesn't use it explicitly for non-equal codes)
{
  // Current internal node index this thread is responsible for constructing.
  // Internal nodes are indexed 0 to (num_primitives - 2).
  unsigned int internal_node_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (internal_node_idx >= num_primitives - 1) return; // There are N-1 internal nodes.

  // Determine the range of *primitives* this internal_node_idx is an ancestor of.
  // The `internal_node_idx` corresponds to the conceptual split between primitive `internal_node_idx`
  // and primitive `internal_node_idx+1` in the sorted list.
  const int2 prim_range = determine_range_adapted(d_sorted_morton_codes, num_primitives, internal_node_idx);

  // Determine where to split the primitive range [prim_range.x, prim_range.y].
  // `split_prim_idx` is the index of the last primitive in the left child's group.
  const int split_prim_idx = find_split_adapted(d_sorted_morton_codes, prim_range.x, prim_range.y, num_primitives);

  // --- Determine children global indices ---
  // Leaf nodes are in d_bvh_nodes at indices [N-1 ... 2N-2].
  // The k-th sorted primitive (0-indexed) corresponds to leaf node at global_idx = (N-1) + k.
  // Internal nodes are in d_bvh_nodes at indices [0 ... N-2].
  // The internal node representing the split between primitive k and k+1 is at global_idx = k.
  unsigned int leaf_node_base_idx = num_primitives - 1;
  unsigned int left_child_global_idx;
  unsigned int right_child_global_idx;

  // Child A (Left) corresponds to primitives [prim_range.x ... split_prim_idx]
  if (split_prim_idx == prim_range.x) 
  { // Left child is a single primitive, so it's a leaf
    left_child_global_idx = leaf_node_base_idx + split_prim_idx;
  } 
  else 
  { // Left child spans multiple primitives, so it's an internal node.
    // This internal node is the one that represents the split *at the end* of the left sub-range.
    // The internal node corresponding to the split *after* primitive `k` is node `k`.
    left_child_global_idx = split_prim_idx;
  }

  // Child B (Right) corresponds to primitives [split_prim_idx + 1 ... prim_range.y]
  if (split_prim_idx + 1 == prim_range.y) 
  { // Right child is a single primitive, so it's a leaf
    right_child_global_idx = leaf_node_base_idx + (split_prim_idx + 1);
  } 
  else 
  { // Right child spans multiple primitives, so it's an internal node.
    // The internal node corresponding to the split *after* primitive `k` is node `k`.
    // Here, the split for the right child's range effectively starts after `split_prim_idx+1`.
    // The internal node that would be the root of this right subtree is `split_prim_idx + 1`.
    right_child_global_idx = split_prim_idx + 1;
  }

  // --- Store children in d_bvh_nodes[internal_node_idx] ---
  LBVHNode& current_internal_node = d_bvh_nodes[internal_node_idx];
  current_internal_node.num_primitives_in_leaf = 0; // Mark as internal node

  current_internal_node.left_child_offset = left_child_global_idx;
  current_internal_node.right_child_offset = right_child_global_idx;


  // Populate the parent indices array
  d_parent_indices[left_child_global_idx] = internal_node_idx;
  d_parent_indices[right_child_global_idx] = internal_node_idx;
  
  if (internal_node_idx == 0) 
  { // Root node
    d_parent_indices[internal_node_idx] = -1; // Mark root's parent as invalid/special
  }
}

__global__ void set_aabb_kernel_adapted(
  LBVHNode* d_bvh_nodes,
  const int* d_parent_indices,
  const PrimitiveReference* d_primitive_refs, // From RawConfig.d_primitive_references (sorted)
  const Sphere* d_all_spheres,                // From RawConfig.d_all_spheres
  const Triangle* d_all_triangles,            // From RawConfig.d_all_triangles
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
      d_all_spheres,
      d_all_triangles);

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
      config.d_lbvh_nodes,
      d_parent_indices,
      config.d_morton_codes, // Assumed to be sorted
      N,                     // num_primitives (num_leaf_nodes)
      morton_bits
  );
  CUDA_CHECK(cudaGetLastError());

  // --- Stage 3: Calculate AABBs for all nodes (leaves then bottom-up for internal) ---
  // This kernel processes from leaves up
  int grid_dim_aabb = (num_leaf_nodes - 1) / threads_per_block + 1;
  set_aabb_kernel_adapted<<<grid_dim_aabb, threads_per_block>>>(
      config.d_lbvh_nodes,
      d_parent_indices,
      config.d_primitive_references, // Assumed to be sorted by Morton code
      config.d_all_spheres,
      config.d_all_triangles,
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

  // printf("LBVH Build (Karas algorithm) complete. Total nodes: %u\n", config.num_lbvh_nodes);
}
