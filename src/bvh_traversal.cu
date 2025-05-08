#include "bvh_traversal.cuh"
#include "vec3.cuh"     // For float3 operations, fminf, fmaxf, etc.
#include "interval.cuh" // For AABB structure
#include "helper.cuh"   // For getBarycentric if needed, though Triangle::checkObject handles it

// Define a stack size for BVH traversal. Max BVH depth is related to log(N).
// 64 should be sufficient for any reasonable number of primitives.
#define TRAVERSAL_STACK_SIZE 64


__device__ __forceinline__ bool hit_aabb_adapted(
  const AABB& node_bbox,
  const vec3& origin,
  const vec3& inv_dir,
  float ray_t_min,
  float current_max_t) // Renamed from 'length' for clarity
{
  // X slab
  float tx1 = (node_bbox.x.min - origin.x) * inv_dir.x;
  float tx2 = (node_bbox.x.max - origin.x) * inv_dir.x;
  float t_near_x = fminf(tx1, tx2);
  float t_far_x  = fmaxf(tx1, tx2);

  // Y slab
  float ty1 = (node_bbox.y.min - origin.y) * inv_dir.y;
  float ty2 = (node_bbox.y.max - origin.y) * inv_dir.y;
  float t_near_y = fminf(ty1, ty2);
  float t_far_y  = fmaxf(ty1, ty2);

  // Z slab
  float tz1 = (node_bbox.z.min - origin.z) * inv_dir.z;
  float tz2 = (node_bbox.z.max - origin.z) * inv_dir.z;
  float t_near_z = fminf(tz1, tz2);
  float t_far_z  = fmaxf(tz1, tz2);

  // Overall intersection interval with the box slabs
  float t_enter = fmaxf(fmaxf(t_near_x, t_near_y), t_near_z);
  float t_exit  = fminf(fminf(t_far_x, t_far_y), t_far_z);

  // Check if the ray's valid interval [ray_t_min, current_max_t] overlaps
  // with the box's intersection interval [t_enter, t_exit].
  // Also ensure t_enter < t_exit for a valid box intersection.
  return t_enter < t_exit && t_enter < current_max_t && t_exit > ray_t_min;
}


__device__ bool intersect_leaf_primitives(
  const LBVHNode& leaf_node,
  const Ray& input_ray, // Use a const ref for input ray
  const RawConfig* config,
  ObjectInfo& current_closest_hit, // Pass by reference to update
  float& current_t_max)            // Pass t_max by reference
{
  bool found_closer_hit_in_leaf = false;
  // Your current LBVHNode stores one primitive per leaf via primitive_offset
  // and num_primitives_in_leaf = 1.

  unsigned int primitive_ref_idx = leaf_node.primitive_offset;
  const PrimitiveReference& prim_ref = config->d_primitive_references[primitive_ref_idx];

  ObjectInfo hit_info_this_primitive; // To store result from checkObject

  if (prim_ref.type == PrimitiveType::SPHERE) 
  {
    hit_info_this_primitive = checkSphereIntersectionSoA(
                                input_ray,
                                prim_ref.id_in_type_array,
                                *config->d_spheres_soa,
                                config);
  } 
  else if (prim_ref.type == PrimitiveType::TRIANGLE) 
  {
    hit_info_this_primitive = checkTriangleIntersectionSoA(
                                input_ray,
                                prim_ref.id_in_type_array,
                                *config->d_triangles_soa,
                                config);
  }

  if (hit_info_this_primitive.isHit && hit_info_this_primitive.distance > 1e-6f && /* ray_t_min usually 0 or epsilon */
      hit_info_this_primitive.distance < current_t_max) 
  {
    current_t_max = hit_info_this_primitive.distance;
    current_closest_hit = hit_info_this_primitive; // Update the best hit so far
    found_closer_hit_in_leaf = true;
  }

  return found_closer_hit_in_leaf;
}


__device__ ObjectInfo traverse_lbvh(
  const Ray& ray,
  const RawConfig* config,
  float initial_t_max)
{
  ObjectInfo closest_hit_info; // Default constructor sets isHit = false, distance = -1.0f
  closest_hit_info.distance = initial_t_max; // Use initial_t_max as current t_max for comparisons
  float current_t_max_for_intersections = initial_t_max;

  if (config->d_lbvh_nodes == nullptr || config->num_lbvh_nodes == 0) 
  {
    return closest_hit_info; // No BVH to traverse
  }

  vec3 inv_dir = vec3(1.0f / ray.dir.x, 1.0f / ray.dir.y, 1.0f / ray.dir.z);
  float ray_t_min = 0.0001f; // Small epsilon to avoid self-intersection at origin

  // Traversal stack storing global indices of nodes to visit
  unsigned int stack[TRAVERSAL_STACK_SIZE];
  int stack_ptr = 0;

  // Start traversal from the root node (assuming root is at index 0)
  unsigned int current_node_idx = 0; // Root node index

  // Push a sentinel value or handle empty stack appropriately.
  // The reference pushes NULL. Here, we can use a large invalid index or manage stack_ptr.
  // For simplicity, the loop condition will handle stack_ptr.

  while (true) 
  {
    const LBVHNode& node = config->d_lbvh_nodes[current_node_idx];

    if (node.isLeaf()) 
    {
      // Intersect with primitives in this leaf node
      intersect_leaf_primitives(node, ray, config, closest_hit_info, current_t_max_for_intersections);

      // Pop from stack if not empty
      if (stack_ptr == 0) break; // Stack is empty, traversal done
      current_node_idx = stack[--stack_ptr]; // Pop
      continue; // Continue with the popped node
    }

    // Internal node: test children
    unsigned int child_l_idx = node.left_child_offset;
    unsigned int child_r_idx = node.right_child_offset;

    // It's good practice to check if child indices are valid, though a correct BVH should ensure this.
    // bool valid_l_child = child_l_idx < config->num_lbvh_nodes;
    // bool valid_r_child = child_r_idx < config->num_lbvh_nodes;

    const LBVHNode& child_l_node = config->d_lbvh_nodes[child_l_idx];
    const LBVHNode& child_r_node = config->d_lbvh_nodes[child_r_idx];

    bool hit_l = hit_aabb_adapted(child_l_node.bbox, ray.eye, inv_dir, ray_t_min, current_t_max_for_intersections);
    bool hit_r = hit_aabb_adapted(child_r_node.bbox, ray.eye, inv_dir, ray_t_min, current_t_max_for_intersections);

    if (hit_l && hit_r) 
    {
      // Both children hit. Heuristic: traverse the closer one first (optional, can be fixed order).
      // For simplicity, traverse left, push right.
      current_node_idx = child_l_idx;
      if (stack_ptr < TRAVERSAL_STACK_SIZE) 
      { // Basic stack overflow check
        stack[stack_ptr++] = child_r_idx; // Push right child
      } 
      else 
      {
        // Stack overflow - extremely deep/bad BVH or too small stack.
        // This should ideally not happen with a reasonably sized stack.
        // Consider error handling or just proceeding with one child.
        printf("Warning: BVH traversal stack overflow!\n");
      }
    } 
    else if (hit_l) 
    {
      current_node_idx = child_l_idx; // Traverse only left
    } 
    else if (hit_r) 
    {
      current_node_idx = child_r_idx; // Traverse only right
    } 
    else 
    {
      // Neither child hit, pop from stack
      if (stack_ptr == 0) break; // Stack is empty, traversal done
      current_node_idx = stack[--stack_ptr]; // Pop
    }
  } // end while(true)

  return closest_hit_info;
}