#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <climits> 

#include "struct.hpp"
#include "object.hpp"
#include "vec3.hpp"

class Config 
{
public:
  Config() : 
    forward(0.0, 0.0, -1.0), right(1.0, 0.0, 0.0),
    up(0.0, 1.0, 0.0), eye{0.0, 0.0, 0.0}, target_up(0.0, 1.0, 0.0) 
    {}

  ~Config() 
  {
    // need to manually remove objects and bvh_head
    for (auto obj : objects)
    {
      delete(obj);
    }
    
    delete(bvh_head);
  }

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
  std::vector<Object*> objects = {};
  Object* bvh_head = nullptr;
  std::vector<Sun> sun = {};
  std::vector<Bulb> bulbs = {};
  std::vector<Plane> planes = {};
  std::vector<Vertex> vertices;
  std::vector<Triangle> triangles;
};

#endif // CONFIG_HPP