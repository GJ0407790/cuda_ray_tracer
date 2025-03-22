#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <climits>
#include <memory>
#include <vector>

#include "struct.cuh"
#include "object.cuh"
#include "vec3.cuh"

class Object;
class Sun;
class Bulb;
class Plane;
class Vertex;

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

  std::vector<Object*> objects;
  Object* bvh_head = nullptr;
  std::vector<Sun*> sun;
  std::vector<Bulb*> bulbs;
  std::vector<Plane*> planes;
  std::vector<Vertex*> vertices;
};

// Same as Config but without STL containers
struct RawConfig {
  int width;
  int height;
  RGB color;
  int bounces;
  int aa;
  double dof_focus;
  double dof_lens;

  vec3 forward;
  vec3 right;
  vec3 up;
  point3 eye;
  vec3 target_up;

  double expose;
  bool fisheye;
  bool panorama;

  double ior;
  double rough;
  int gi;
  RGB trans;
  RGB shine;

  // Pointers to device memory
  Object* bvh_head;
  int num_sun;
  Sun* sun;
  int num_bulbs;
  Bulb* bulbs;
  int num_planes;
  Plane* planes;
};

#endif // CONFIG_HPP
