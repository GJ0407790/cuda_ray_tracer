/**
 * @file keywords.hpp
 * @author Jin (jinj2@illinois.edu)
 * @brief 
 * @version 0.1
 * @date 2024-11-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef KEYWORDS_H
#define KEYWORDS_H
#include "helper.hpp"
#include "struct.hpp"
#include "vec3.hpp"
#include <string>
#include <vector>

//width and height of output image

extern int width;
extern int height;
extern std::string filename;
extern int bounces;
extern int aa;
extern int gi;
extern double dof_focus;
extern double dof_lens;
extern vec3 forward;
extern vec3 right;
extern vec3 up;
extern point3 eye;
extern double expose;
extern bool fisheye;
extern bool panorama;
extern std::string texture;
//sphere: a vector of [x,y,z,r]
extern shared_ptr<Object> bvh_head;
extern std::vector<Sun> sun;
extern std::vector<Bulb> bulbs;
extern std::vector<Plane> planes;
extern std::vector<Vertex> vertices;
extern std::vector<Triangle> triangles;

#endif
