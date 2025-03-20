#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <climits>
#include <memory>
#include <vector>

#include "struct.hpp"
#include "object.hpp"
#include "vec3.cuh"

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
  double dof_focus = 0;
  double dof_lens = 0;

  vec3 forward;
  vec3 right;
  vec3 up;
  point3 eye;
  vec3 target_up;

  double expose = INT_MAX;
  bool fisheye = false;
  bool panorama = false;

  double ior = 1.458;
  double rough = 0;
  int gi = 0;
  RGB trans;
  RGB shine;

  std::string texture = "none";
  Texcoord texcoord;
  std::vector<std::shared_ptr<Object>> objects = {};
  std::shared_ptr<Object> bvh_head = nullptr;
  std::vector<Sun> sun = {};
  std::vector<Bulb> bulbs = {};
  std::vector<Plane> planes = {};
  std::vector<Vertex> vertices;
  std::vector<Triangle> triangles;
};

// Same as Config but without STL containers
class RawConfig 
{
public:
	RawConfig(StlConfig& config)
  {
    width = config.width;
    height = config.height;
    bounces = config.bounces;
    aa = config.aa;
    dof_focus = config.dof_focus;
    dof_lens = config.dof_lens;
    expose = config.expose;
    fisheye = config.fisheye;
    panorama = config.panorama;
    ior = config.ior;
    rough = config.rough;
    gi = config.gi;
    color = config.color;
    forward = config.forward;
    right = config.right;
    up = config.up;
    eye = config.eye;
    target_up = config.target_up;
    
    num_objects = config.objects.size();
    if (num_objects > 0) {
      objects = new Object*[num_objects];  // Allocate new array
      for (size_t i = 0; i < num_objects; i++) {
        objects[i] = config.objects[i].get();  
      }
    } else {
      objects = nullptr;
    }

    bvh_head = config.bvh_head ? config.bvh_head.get() : nullptr;

    num_sun = config.sun.size();
    if (num_sun > 0) {
      sun = new Sun[num_sun];
      for (size_t i = 0; i < num_sun; i++) {
        sun[i] = config.sun[i];
      }
    } else {
      sun = nullptr;
    }

    num_bulbs = config.bulbs.size();
    if (num_bulbs > 0) {
      bulbs = new Bulb[num_bulbs];
      for (size_t i = 0; i < num_bulbs; i++) {
        bulbs[i] = config.bulbs[i];
      }
    } else {
      bulbs = nullptr;
    }

    num_planes = config.planes.size();
    if (num_planes > 0) {
      planes = new Plane[num_planes];
      for (size_t i = 0; i < num_planes; i++) {
        planes[i] = config.planes[i];
      }
    } else {
      planes = nullptr;
    }

    num_vertices = config.vertices.size();
    if (num_vertices > 0) {
      vertices = new Vertex[num_vertices];
      for (size_t i = 0; i < num_vertices; i++) {
        vertices[i] = config.vertices[i];
      }
    } else {
      vertices = nullptr;
    }

    num_triangles = config.triangles.size();
    if (num_triangles > 0) {
      triangles = new Triangle[num_triangles];
      for (size_t i = 0; i < num_triangles; i++) {
        triangles[i] = config.triangles[i];
      }
    } else {
      triangles = nullptr;
    }
  }

  ~RawConfig() {
    delete[] objects;
    delete[] sun;
    delete[] bulbs;
    delete[] planes;
    delete[] vertices;
    delete[] triangles;
  }

public:
  int width = 0;
  int height = 0;
  RGB color = {1.0,1.0,1.0};
  int bounces = 4;
  int aa = 0;
  double dof_focus = 0;
  double dof_lens = 0;

  vec3 forward;
  vec3 right;
  vec3 up;
  point3 eye;
  vec3 target_up;

  double expose = INT_MAX;
  bool fisheye = false;
  bool panorama = false;

  double ior = 1.458;
  double rough = 0;
  int gi = 0;
  RGB trans;
  RGB shine;

  std::string texture = "none";
  Texcoord texcoord;

  // Scene Data (Raw Pointers)
  int num_objects = 0;
  Object** objects = nullptr;  // Needs manual deallocation
  Object* bvh_head = nullptr;
  
  int num_sun = 0;
  Sun* sun = nullptr;
  
  int num_bulbs = 0;
  Bulb* bulbs = nullptr;
  
  int num_planes = 0;
  Plane* planes = nullptr;
  
  int num_vertices = 0;
  Vertex* vertices = nullptr;
  
  int num_triangles = 0;
  Triangle* triangles = nullptr;
};

#endif // CONFIG_HPP
