#include "config.hpp"
#include "helper.cuh"
#include "parse.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>

using std::string;
using std::vector;
using std::shared_ptr;

void parseInput(char* argv[], StlConfig& config)
{
	std::string filePath = argv[1];
	
	std::ifstream input(filePath);
	
	if(!input){
		std::cout <<  "Error opening file..."  << std::endl;
		exit(1);
	}

	//read line by line
	AABB running_scene_bounds;
	string line;
	while(getline(input,line)){
		std::stringstream ss(line);
		std::vector<std::string> words;
		string word;
		while (ss >> word) words.push_back(word);
		parseLine(words, config, running_scene_bounds);
	}


}

void parseLine(std::vector<std::string> words, StlConfig& config, AABB& running_scene_bounds)
{
	//return on empty line
	if(words.empty()) return;
	
	//"png"
	if(words[0] == "png" && words.size() == 4){
		config.width = stoi(words[1]);
		config.height = stoi(words[2]);
		config.filename = words[3];
	}
	/*------------*/
	/*Mode setting*/
	/*------------*/

	/*Bounces, the number of times a ray will bounce before it stops*/
	else if(words[0] == "bounces" && words.size() == 2){
		config.bounces = stoi(words[1]);
	}
	/*The "forward" direction.*/
	else if(words[0] == "forward" && words.size() == 4){
		config.forward = {stof(words[1]),stof(words[2]),stof(words[3])};
		config.right = cross(config.forward, config.up).normalize();
		config.up = cross(config.right, config.forward).normalize();
	}
	/*The target "up" direction, but not the real one*/
	else if(words[0] == "up" && words.size() == 4){
		config.target_up = {stof(words[1]),stof(words[2]),stof(words[3])};
		config.right = cross(config.forward, config.target_up).normalize();
		config.up = cross(config.right, config.forward).normalize();
	}
	/*Eye location, the ray origin for primary rays*/
	else if(words[0] == "eye" && words.size() == 4){
		config.eye = {stof(words[1]),stof(words[2]),stof(words[3])};
	}
	/*Exposure*/
	else if(words[0] == "expose" && words.size() == 2){
		config.expose = stof(words[1]);
	}
	/*Depth of field*/
	else if(words[0] == "dof" && words.size() == 3){
		config.dof_focus = stof(words[1]); 
		config.dof_lens = stof(words[2]);
	}
	/*Anti-aliasing*/
	else if(words[0] == "aa" && words.size() == 2){
		config.aa = stoi(words[1]);
	}
	/*Panorama view*/
	else if(words[0] == "panorama" && words.size() == 1){
		config.panorama = true;
	}
	/*Fisheye view*/
	else if(words[0] == "fisheye" && words.size() == 1){
		config.fisheye = true;
	}
	else if(words[0] == "gi" && words.size() == 2){
		config.gi = stoi(words[1]);
	}
	/*-------------*/
	/*State setting*/
	/*-------------*/
	else if(words[0] == "color" && words.size() == 4){
		float r,g,b;
		r = stof(words[1]);g = stof(words[2]);
		b = stof(words[3]);
		config.color = {r,g,b};
	}
	else if(words[0] == "roughness" && words.size() == 2){
		config.rough = stof(words[1]);
	}
	else if(words[0] == "shininess" && words.size() == 2){
		config.shine = {stof(words[1]),stof(words[1]),stof(words[1])};
	}
	else if(words[0] == "shininess" && words.size() == 4){
		config.shine = {stof(words[1]),stof(words[2]),stof(words[3])};
	}
	else if(words[0] == "transparency" && words.size() == 2){
		config.trans = {stof(words[1]),stof(words[1]),stof(words[1])};
	}
	else if(words[0] == "transparency" && words.size() == 4){
		config.trans = {stof(words[1]),stof(words[2]),stof(words[3])};
	}
	else if(words[0] == "ior" && words.size() == 2){
		config.ior = stof(words[1]);
	}
	/*----------------*/
	/*Geometry setting*/
	/*----------------*/
	else if(words[0] == "sphere" && words.size() == 5){
		float x,y,z,r;

		x = stof(words[1]);y = stof(words[2]);
		z = stof(words[3]);r = stof(words[4]);

		config.host_spheres_data.emplace_back(x, y, z, r, config.color);

		auto& sphere = config.host_spheres_data.back();

		sphere.mat.shininess = config.shine;
		sphere.mat.trans = config.trans;
		sphere.mat.ior = config.ior;
		sphere.mat.roughness = config.rough;

		config.host_primitive_references.emplace_back(PrimitiveType::SPHERE, config.host_spheres_data.size() - 1);

		auto rvec = vec3(sphere.r, sphere.r, sphere.r);
		const auto& prim_bbox = AABB(sphere.c - rvec, sphere.c + rvec);

		running_scene_bounds.x.min = fminf(running_scene_bounds.x.min, prim_bbox.x.min);
		running_scene_bounds.x.max = fmaxf(running_scene_bounds.x.max, prim_bbox.x.max);
		running_scene_bounds.y.min = fminf(running_scene_bounds.y.min, prim_bbox.y.min);
		running_scene_bounds.y.max = fmaxf(running_scene_bounds.y.max, prim_bbox.y.max);
		running_scene_bounds.z.min = fminf(running_scene_bounds.z.min, prim_bbox.z.min);
		running_scene_bounds.z.max = fmaxf(running_scene_bounds.z.max, prim_bbox.z.max);
	}
	else if(words[0] == "plane" && words.size() == 5){
		float a,b,c,d;
		a = stof(words[1]);b = stof(words[2]);
		c = stof(words[3]);d = stof(words[4]);

		config.plane_data.emplace_back(a,b,c,d,config.color);
		config.plane_data.back().setProperties(config.shine,config.trans,config.ior,config.rough);
	}
	/*Vertices*/
	else if(words[0] == "xyz" && words.size() == 4){
		float x,y,z;
		x = stof(words[1]);y = stof(words[2]);
		z = stof(words[3]);

		config.vertex_data.emplace_back(x,y,z);
	}
	/*Triangle index*/
	else if(words[0] == "tri" && words.size() == 4){
		float i,j,k;
		int size = config.vertex_data.size();

		i = (stoi(words[1]) > 0) ? stoi(words[1]) - 1 : size + stoi(words[1]);
		j = (stoi(words[2]) > 0) ? stoi(words[2]) - 1 : size + stoi(words[2]);
		k = (stoi(words[3]) > 0) ? stoi(words[3]) - 1 : size + stoi(words[3]);

		config.host_triangles_data.emplace_back(config.vertex_data[i], config.vertex_data[j], config.vertex_data[k], config.color);

		auto& tri = config.host_triangles_data.back();

		tri.mat.shininess = config.shine;
		tri.mat.trans = config.trans;
		tri.mat.ior = config.ior;
		tri.mat.roughness = config.rough;

		config.host_primitive_references.emplace_back(PrimitiveType::TRIANGLE, config.host_triangles_data.size() - 1);

		const auto& prim_bbox = AABB(tri.p0, tri.p1, tri.p2);

		running_scene_bounds.x.min = fminf(running_scene_bounds.x.min, prim_bbox.x.min);
		running_scene_bounds.x.max = fmaxf(running_scene_bounds.x.max, prim_bbox.x.max);
		running_scene_bounds.y.min = fminf(running_scene_bounds.y.min, prim_bbox.y.min);
		running_scene_bounds.y.max = fmaxf(running_scene_bounds.y.max, prim_bbox.y.max);
		running_scene_bounds.z.min = fminf(running_scene_bounds.z.min, prim_bbox.z.min);
		running_scene_bounds.z.max = fmaxf(running_scene_bounds.z.max, prim_bbox.z.max);
	}
	else if(words[0] == "sun" && words.size() == 4){
		float x,y,z;
		x = stof(words[1]);y = stof(words[2]);
		z = stof(words[3]);
		
		config.sun_data.emplace_back(x,y,z,config.color);
	}
	/*Light bulb, a point of light in the scene*/
	else if(words[0] == "bulb" && words.size() == 4){
		float x,y,z;
		x = stof(words[1]);y = stof(words[2]);
		z = stof(words[3]);

		config.bulb_data.emplace_back(x,y,z,config.color);
	}
	/*Fail case*/
	else{
		std::cout << "One of the lines are not valid." << std::endl;
		exit(1);
	}
}
