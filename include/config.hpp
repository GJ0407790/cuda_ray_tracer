#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <climits>
#include <memory>
#include <vector>

#include "object.cuh" // For AABB, Sphere, Triangle, PrimitiveReference, etc.
#include "lbvh.cuh"
#include "vec3.cuh"

// Forward declarations (if Object class is not fully defined yet or for BVHNode later)
class Sun;
class Bulb;
class Plane;
class Vertex;
class Triangle;
class Sphere;
struct PrimitiveReference;
struct LBVHNode;
class Materials;
class SphereDataSoA;
class TriangleDataSoA;

class StlConfig
{
public:
  StlConfig() :
    forward(0.0, 0.0, -1.0), right(1.0, 0.0, 0.0),
    up(0.0, 1.0, 0.0), eye{0.0, 0.0, 0.0}, target_up(0.0, 1.0, 0.0)
  {}

public:
  int width = 0;
  int height = 0;
  std::string filename = "file.txt";
  RGB color = {1.0,1.0,1.0};
  int bounces = 4;
  int aa = 0;
  float dof_focus = 0.0f;
  float dof_lens = 0.0f;

  vec3 forward;
  vec3 right;
  vec3 up;
  point3 eye;
  vec3 target_up;

  float expose = float(INFINITY);
  bool fisheye = false;
  bool panorama = false;

  float ior = 1.458f;
  float rough = 0.0f;
  int gi = 0;
  RGB trans;
  RGB shine;

  // Host-side storage for parsed objects

  // Host-side SoA data and primitive references (populated from 'objects')
  std::vector<Sphere> host_spheres_data;
  std::vector<Triangle> host_triangles_data;
  std::vector<PrimitiveReference> host_primitive_references;
  AABB scene_bounds_host; // Calculated on host after parsing

  std::vector<Sun> sun_data;
  std::vector<Bulb> bulb_data;
  std::vector<Plane> plane_data;
  std::vector<Vertex> vertex_data; // Used during parsing then can be cleared
};

// Same as Config but without STL containers and for device
struct RawConfig {
  int width;
  int height;
  RGB color;
  int bounces;
  int aa;
  float dof_focus;
  float dof_lens;

  vec3 forward;
  vec3 right;
  vec3 up;
  point3 eye;
  vec3 target_up;

  float expose;
  bool fisheye;
  bool panorama;

  float ior;    // These might become per-primitive if materials are complex
  float rough;
  int gi;
  RGB trans;
  RGB shine;

  // Device pointers for SoA primitive data
  SphereDataSoA* d_spheres_soa;
  int num_spheres;
  TriangleDataSoA* d_triangles_soa;
  int num_triangles;

  // Device array for primitive references (this one gets sorted by Morton code)
  PrimitiveReference* d_primitive_references;
  // Device array for Morton codes (sorted along with d_primitive_references)
  unsigned int* d_morton_codes;
  int num_total_primitives; // num_spheres + num_triangles

  float3 scene_min_corner;
  float3 scene_max_corner;

  // --- LBVH Data ---
  LBVHNode* d_lbvh_nodes;
  int num_lbvh_nodes; // Total nodes in the BVH (2 * num_leaf_nodes - 1, or similar)

  // Pointers to device memory for lights, planes etc. (as before)
  int num_sun;
  Sun* d_all_suns;
  int num_bulbs;
  Bulb* d_all_bulbs;
  int num_planes;
  Plane* d_all_planes;
};

#endif // CONFIG_HPP