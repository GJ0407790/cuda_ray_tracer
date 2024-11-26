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
#include "../include/all.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <climits>
#include <memory>
using std::string;
using std::vector;
using std::shared_ptr;
//setting keywords
int width = 0;
int height = 0;
std::string filename = "file.txt";
RGB color = {1.0,1.0,1.0};
int bounces = 4;
int aa = 0;
double dof_focus = 0;
double dof_lens = 0;
vec3 forward(0.0,0.0,-1.0);
vec3 right(1.0,0.0,0.0);
vec3 up(0.0,1.0,0.0);
point3 eye = {0.0,0.0,0.0};
vec3 target_up(0.0,1.0,0.0);
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
std::vector<shared_ptr<Object>> objects = {};
shared_ptr<Object> bvh_head = nullptr;
std::vector<Sun> sun = {};
std::vector<Bulb> bulbs = {};
std::vector<Plane> planes = {};
std::vector<Vertex> vertices;
std::vector<Triangle> triangles;


void parseInput(char* argv[]){
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
		parseLine(words);
	}

	bvh_head = std::make_shared<BVH>(objects,0,objects.size(),0);
}

void parseLine(std::vector<std::string> words){
	//return on empty line
	if(words.empty()) return;
	
	//"png"
	if(words[0] == "png" && words.size() == 4){
		width = stoi(words[1]);
		height = stoi(words[2]);
		filename = words[3];
	}
	/*------------*/
	/*Mode setting*/
	/*------------*/

	/*Bounces, the number of times a ray will bounce before it stops*/
	else if(words[0] == "bounces" && words.size() == 2){
		bounces = stoi(words[1]);
	}
	/*The "forward" direction.*/
	else if(words[0] == "forward" && words.size() == 4){
		forward = {stof(words[1]),stof(words[2]),stof(words[3])};
		right = cross(forward,up).normalize();
		up = cross(right,forward).normalize();
	}
	/*The target "up" direction, but not the real one*/
	else if(words[0] == "up" && words.size() == 4){
		target_up = {stof(words[1]),stof(words[2]),stof(words[3])};
		right = cross(forward,target_up).normalize();
		up = cross(right,forward).normalize();
	}
	/*Eye location, the ray origin for primary rays*/
	else if(words[0] == "eye" && words.size() == 4){
		eye = {stof(words[1]),stof(words[2]),stof(words[3])};
	}
	/*Exposure*/
	else if(words[0] == "expose" && words.size() == 2){
		expose = stof(words[1]);
	}
	/*Depth of field*/
	else if(words[0] == "dof" && words.size() == 3){
		dof_focus = stof(words[1]);dof_lens = stof(words[2]);
	}
	/*Anti-aliasing*/
	else if(words[0] == "aa" && words.size() == 2){
		aa = stoi(words[1]);
	}
	/*Panorama view*/
	else if(words[0] == "panorama" && words.size() == 1){
		panorama = true;
	}
	/*Fisheye view*/
	else if(words[0] == "fisheye" && words.size() == 1){
		fisheye = true;
	}
	else if(words[0] == "gi" && words.size() == 2){
		gi = stoi(words[1]);
	}
	/*-------------*/
	/*State setting*/
	/*-------------*/
	else if(words[0] == "color" && words.size() == 4){
		double r,g,b;
		r = stof(words[1]);g = stof(words[2]);
		b = stof(words[3]);
		color = {r,g,b};
	}
	else if(words[0] == "texcoord" && words.size() == 3){
		texcoord = {stof(words[1]),stof(words[2])};
	}
	else if(words[0] == "texture" && words.size() == 2){
		texture = words[1];
	}
	else if(words[0] == "roughness" && words.size() == 2){
		rough = stof(words[1]);
	}
	else if(words[0] == "shininess" && words.size() == 2){
		shine = {stof(words[1]),stof(words[1]),stof(words[1])};
	}
	else if(words[0] == "shininess" && words.size() == 4){
		shine = {stof(words[1]),stof(words[2]),stof(words[3])};
	}
	else if(words[0] == "transparency" && words.size() == 2){
		trans = {stof(words[1]),stof(words[1]),stof(words[1])};
	}
	else if(words[0] == "transparency" && words.size() == 4){
		trans = {stof(words[1]),stof(words[2]),stof(words[3])};
	}
	else if(words[0] == "ior" && words.size() == 2){
		ior = stof(words[1]);
	}
	/*----------------*/
	/*Geometry setting*/
	/*----------------*/
	else if(words[0] == "sphere" && words.size() == 5){
		double x,y,z,r;shared_ptr<Object> s;
		x = stof(words[1]);y = stof(words[2]);
		z = stof(words[3]);r = stof(words[4]);
		if(texture != "none")s = std::make_shared<Sphere>(x,y,z,r,texture);
		else s = std::make_shared<Sphere>(x,y,z,r,color);
		s->setProperties(shine,trans,ior,rough);
		objects.push_back(s);
	}
	else if(words[0] == "plane" && words.size() == 5){
		double a,b,c,d;
		a = stof(words[1]);b = stof(words[2]);
		c = stof(words[3]);d = stof(words[4]);
		Plane p(a,b,c,d,color);
		p.setProperties(shine,trans,ior,rough);
		planes.push_back(p);
	}
	/*Vertices*/
	else if(words[0] == "xyz" && words.size() == 4){
		double x,y,z;
		x = stof(words[1]);y = stof(words[2]);
		z = stof(words[3]);
		Vertex vert(x,y,z,texcoord);
		vertices.push_back(vert);
	}
	/*Triangle index*/
	else if(words[0] == "tri" && words.size() == 4){
		double i,j,k;
		int size = vertices.size();
		shared_ptr<Object> t;
		i = (stoi(words[1]) > 0) ? stoi(words[1]) - 1 : size + stoi(words[1]);
		j = (stoi(words[2]) > 0) ? stoi(words[2]) - 1 : size + stoi(words[2]);
		k = (stoi(words[3]) > 0) ? stoi(words[3]) - 1 : size + stoi(words[3]);
		if(texture != "none")t = std::make_shared<Triangle>(vertices[i],vertices[j],vertices[k],texture);
		else t = std::make_shared<Triangle>(vertices[i],vertices[j],vertices[k],color);
		t->setProperties(shine,trans,ior,rough); //might break texture here!
		objects.push_back(t);
	}
	else if(words[0] == "sun" && words.size() == 4){
		double x,y,z;
		x = stof(words[1]);y = stof(words[2]);
		z = stof(words[3]);
		Sun s(x,y,z,color);
		sun.push_back(s);
	}
	/*Light bulb, a point of light in the scene*/
	else if(words[0] == "bulb" && words.size() == 4){
		double x,y,z;
		x = stof(words[1]);y = stof(words[2]);
		z = stof(words[3]);
		Bulb b(x,y,z,color);
		bulbs.push_back(b);
	}
	/*Fail case*/
	else{
		printErr("One of the lines are not valid.");
		exit(1);
	}

}
