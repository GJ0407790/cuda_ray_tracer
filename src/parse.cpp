/**
 * @file parse.cpp
 * @author Jin (jinj2@illinois.edu)
 * @brief 
 * @version 0.1
 * @date 2024-11-06
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "../include/config.hpp"
#include "../include/helper.hpp"
#include "../include/parse.hpp"

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <vector>

using std::string;
using std::vector;
using std::shared_ptr;

extern RawConfig config;

void parseInput(char* argv[], StlConfig& config){
	std::string filePath = argv[1];
	
	//The grading server and my local environment have different input locations
	std::ifstream input(filePath);
	if (!input.is_open())input.open("input/" + filePath);
	//Check if the file opened successfully
	if(!input){
		printErr("Error opening file " + filePath);
		printErr("Check if the file exists.");
		exit(1);
	}

	//read line by line
	string line;
	while(getline(input,line)){
		std::stringstream ss(line);
		std::vector<std::string> words;
		string word;
		while (ss >> word) words.push_back(word);
		parseLine(words, config);
	}

	config.bvh_head = std::make_shared<BVH>(config.objects, 0, config.objects.size(), 0);
}

void parseLine(std::vector<std::string> words, StlConfig& config){
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
		double r,g,b;
		r = stof(words[1]);g = stof(words[2]);
		b = stof(words[3]);
		config.color = {r,g,b};
	}
	else if(words[0] == "texcoord" && words.size() == 3){
		config.texcoord = {stof(words[1]),stof(words[2])};
	}
	else if(words[0] == "texture" && words.size() == 2){
		config.texture = words[1];
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
		double x,y,z,r;
		shared_ptr<Object> s;

		x = stof(words[1]);y = stof(words[2]);
		z = stof(words[3]);r = stof(words[4]);

		if(config.texture != "none")
		{
			s = std::make_shared<Sphere>(x,y,z,r,config.texture);
		}
		else
		{ 
			s = std::make_shared<Sphere>(x,y,z,r,config.color);
		}

		s->setProperties(config.shine, config.trans, config.ior, config.rough);
		config.objects.push_back(s);
	}
	else if(words[0] == "plane" && words.size() == 5){
		double a,b,c,d;
		a = stof(words[1]);b = stof(words[2]);
		c = stof(words[3]);d = stof(words[4]);
		Plane p(a,b,c,d,config.color);
		p.setProperties(config.shine,config.trans,config.ior,config.rough);
		config.planes.push_back(std::move(p));
	}
	/*Vertices*/
	else if(words[0] == "xyz" && words.size() == 4){
		double x,y,z;
		x = stof(words[1]);y = stof(words[2]);
		z = stof(words[3]);
		Vertex vert(x,y,z,config.texcoord);
		config.vertices.push_back(std::move(vert));
	}
	/*Triangle index*/
	else if(words[0] == "tri" && words.size() == 4){
		double i,j,k;
		int size = config.vertices.size();
		shared_ptr<Object> t;
		i = (stoi(words[1]) > 0) ? stoi(words[1]) - 1 : size + stoi(words[1]);
		j = (stoi(words[2]) > 0) ? stoi(words[2]) - 1 : size + stoi(words[2]);
		k = (stoi(words[3]) > 0) ? stoi(words[3]) - 1 : size + stoi(words[3]);
		
		if(config.texture != "none")
		{
			t = std::make_shared<Triangle>(config.vertices[i], config.vertices[j], config.vertices[k], config.texture);
		}
		else
		{
			t = std::make_shared<Triangle>(config.vertices[i], config.vertices[j], config.vertices[k], config.color);
		} 

		t->setProperties(config.shine,config.trans,config.ior,config.rough); //might break texture here!
		config.objects.push_back(t);
	}
	else if(words[0] == "sun" && words.size() == 4){
		double x,y,z;
		x = stof(words[1]);y = stof(words[2]);
		z = stof(words[3]);
		Sun s(x,y,z,config.color);
		config.sun.push_back(std::move(s));
	}
	/*Light bulb, a point of light in the scene*/
	else if(words[0] == "bulb" && words.size() == 4){
		double x,y,z;
		x = stof(words[1]);y = stof(words[2]);
		z = stof(words[3]);
		Bulb b(x,y,z,config.color);
		config.bulbs.push_back(std::move(b));
	}
	/*Fail case*/
	else{
		printErr("One of the lines are not valid.");
		exit(1);
	}

}
