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

  float ior;
  float rough;
  int gi;
  RGB trans;
  RGB shine;

  // Pointers to device memory
  Object* bvh_head;
  int num_sun;
  Sun** sun;
  int num_bulbs;
  Bulb** bulbs;
  int num_planes;
  Plane** planes;
};

#endif // CONFIG_HPP
