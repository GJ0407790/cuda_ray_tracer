/**
 * @file object.hpp
 * @author Jin (jinj2@illinois.edu)
 * @brief 
 * @version 0.1
 * @date 2024-11-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef OBJECT_H
#define OBJECT_H
#include "struct.hpp"

#include <algorithm>
#include <memory>


/**
 * @brief Material related variables for objects in the scene.
 * 		  
 */
class Materials{
public:
	RGB color; //!< The color, in RGB format, of the material
	RGB shininess; //!< The shininess of the object, which is related to reflection. 
	RGB trans; 	//! < The transparency of the object, which is related to refraction. 
	double ior = 1.458; //!< Index of refraction. Default to 1.458.
	double roughness = 0; //!< Roughness of the object, Default to zero(none).
	/**
	 * @brief Construct a new Materials object, with every member as default.
	 */
	Materials() {}
	/**
	 * @brief Construct a new Materials object, with inputs.
	 * @param color 
	 * @param shininess 
	 * @param trans 
	 * @param ior 
	 * @param roughness 
	 */
	Materials(RGB color,RGB shininess,RGB trans,double ior,double roughness):
	color(color),shininess(shininess),trans(trans),ior(ior),roughness(roughness) {}
};

/**
 * @brief ObjectInfo is a class that passes the parameters of a ray hit, and the
 * 		details of the object that was hit.
 */
class ObjectInfo{
public:
	bool isHit = false; //!< Signals if there was a hit. Defaults to false.
	double distance; //!< The distance from the ray origin to the object.
	point3 i_point;  //!< The intersection point.
	vec3 normal; //!< The normal at the point of intersection.
	Materials mat; //!< Material properties.
	ObjectInfo(): distance(-1.0),i_point(point3()),normal(vec3()),mat(Materials()) {}
    ObjectInfo(double distance,point3 i_point,
	vec3 normal,Materials mat): 
		isHit(true),distance(distance),i_point(i_point),normal(normal),mat(mat) {}
};

/**
 * @brief The object class, both BVH nodes and objects in the scene.
 */
class Object{
public:
	AABB bbox; //!< Axis-aligned bounding box for the Object class. For BVH traversal.
	virtual ObjectInfo checkObject(Ray& ray) = 0;
	virtual void setProperties(RGB,RGB,double,double){}
	virtual AABB getBox() = 0;
};

/**
 * @brief Class sphere, with one center point 
 *        and a radius, together with a rgb color
 *        value.
 */
class Sphere : public Object{
public:
	point3 c; //!< Center point of the sphere.
	double r; //!< Radius of the sphere.
	Materials mat; //!< Material properties.
	Image texture; //!< The texture, as a image.
	/**
	 * @brief Construct a new Sphere object with no inputs
	 */
	Sphere(): r(0.0) {
		c = point3(0.0,0.0,0.0);
		mat.color = {1.0f,1.0f,1.0f};
	}
	/**
	 * @brief Construct a new Sphere object, with color inputs.
	 */
	Sphere(double x,double y,double z,double r,RGB rgb): r(r) {
		c = point3(x,y,z);
		mat.color = rgb;
		auto rvec = vec3(r,r,r);
        bbox = AABB(c - rvec,c + rvec);
	}
	/**
	 * @brief Construct a new Sphere object, with texture inputs.
	 */
	Sphere(double x,double y,double z,double r,std::string text): r(r) {
		c = point3(x,y,z);
		//texture = load_image(text.c_str());
		texture = Image((text).c_str());
		if (texture.empty()) texture = Image(("input/"+text).c_str());
		auto rvec = vec3(r,r,r);
        bbox = AABB(c - rvec,c + rvec);
	}
    ObjectInfo checkObject(Ray& ray) override;
	/**
	 * @brief Get the Axis-aligned bounding box.
	 * @return AABB Axis-aligned bounding box
	 */
	AABB getBox() override {return bbox;}
	std::tuple<double,double> sphereUV(const point3& point) const;
	RGB getColor(const point3& point);
	/**
	 * @brief Set the Material properties.
	 */
	void setProperties(RGB shine,RGB tran,double ior,double roughness) override {
	mat.shininess = shine;mat.trans = tran;mat.ior = ior;mat.roughness = roughness;
	}
};

/**
 * @brief A plane defined by ax + by + cz + d = 0.
 */
class Plane{
public:
	double a,b,c,d; //!< a,b,c,d in ax + by + cz + d = 0.
	vec3 nor;	//!< Normal of the plane.
	point3 point; //!< A point on the plane, for calculation purposes.
	Materials mat; //!< Material properties.
	Plane(): a(0.0),b(0.0),c(0.0),d(0.0),nor(vec3()),point(point3()) {mat.color = {1.0f,1.0f,1.0f};}
	Plane(double a,double b,double c,double d,RGB rgb): a(a),b(b),c(c),d(d) {
		nor = vec3(a,b,c).normalize();
		point = (-d * vec3(a,b,c)) / (pow(a,2) + pow(b,2) + pow(c,2));
		mat.color = rgb;
	}
	void setProperties(RGB shine,RGB tran,double ior,double roughness){
	mat.shininess = shine;mat.trans = tran;mat.ior = ior;mat.roughness = roughness;
	}
};

/**
 * @brief A Vertex class, used only in input parsing.
 */
class Vertex{
public:
	point3 p; //!< The point for the vertex.
	Texcoord tex; //<! The texture coordinate, if provided.
	Vertex(): p(point3()) {}
	Vertex(double x,double y,double z): p(point3(x,y,z)) {}
	Vertex(double x,double y,double z,Texcoord tex): p(point3(x,y,z)),tex(tex){}
};

/**
 * @brief Class Triangle, with 3 vertices that made it up.
 */
class Triangle : public Object{
public:
	point3 p0,p1,p2; //!< Three vertices of the triangle.
	Texcoord tex0,tex1,tex2; //!<texture coordinates for the vertices,if provided.
	vec3 nor;  //!< Normal of the triangle.
	point3 e1,e2; //!< The e1,e2 coordinates, precomputed for Barycentric calculation.
	Image texture; //!< Texture, if provided.
	Materials mat; //!< Material properties.
	Triangle(): p0(point3()),p1(point3()),p2(point3()),
				nor(vec3()),e1(point3()),e2(point3()){mat.color = {1.0f,1.0f,1.0f};}
	Triangle(Vertex a,Vertex b,Vertex c,RGB rgb) {
		p0 = a.p;p1 = b.p;p2 = c.p;
		tex0 = a.tex;tex1 = b.tex;tex2 = c.tex;
		mat.color = rgb;
		nor = cross(p1-p0,p2-p0).normalize();
		vec3 a1 = cross(p2-p0,nor);
		vec3 a2 = cross(p1-p0,nor);
		e1 = (1/(dot(a1,p1-p0))) * a1;
		e2 = (1/(dot(a2,p2-p0))) * a2;
		bbox = AABB(a.p,b.p,c.p);
	}
	Triangle(Vertex a,Vertex b,Vertex c,std::string text) {
		p0 = a.p;p1 = b.p;p2 = c.p;
		tex0 = a.tex;tex1 = b.tex;tex2 = c.tex;
		nor = cross(p1-p0,p2-p0).normalize();
		vec3 a1 = cross(p2-p0,nor);
		vec3 a2 = cross(p1-p0,nor);
		e1 = (1/(dot(a1,p1-p0))) * a1;
		e2 = (1/(dot(a2,p2-p0))) * a2;
		texture = Image((text).c_str());
		if (texture.empty()) texture = Image(("input/"+text).c_str());
		bbox = AABB(a.p,b.p,c.p);
	}
	ObjectInfo checkObject(Ray& ray) override;
	AABB getBox() override {return bbox;}
	RGB getColor(double b0,double b1,double b2);
	void setProperties(RGB shine,RGB tran,double ior,double roughness) override {
	mat.shininess = shine;mat.trans = tran;mat.ior = ior;mat.roughness = roughness;
	}
};

class Sun{
public:
	vec3 dir;
	RGB color;
	Sun() {
		dir = vec3(0.0,0.0,0.0);
		color = {1.0f,1.0f,1.0f};
	}
	Sun(double x,double y,double z,RGB rgb) {
		dir = vec3(x,y,z);
		color = rgb;
	}
};

class Bulb{
public:
	point3 point;
	RGB color;
	Bulb() {
		point = point3(0.0,0.0,0.0);
		color = {1.0f,1.0f,1.0f};
	}
	Bulb(double x,double y,double z,RGB rgb) {
		point = point3(x,y,z);
		color = rgb;
	}
};


class BVH : public Object{
private:
    shared_ptr<Object> left;
    shared_ptr<Object> right;
public:
    BVH(std::vector<shared_ptr<Object>>& objs,int start,int end,int axis){
        auto comp = (axis == 0) ? box_x_compare
                	: (axis == 1) ? box_y_compare
                    				: box_z_compare;

		int span = end - start;

		if(span == 0){
			return;
		}else if(span == 1){
			left = right = objs[start];
		}else if(span == 2){
			left = objs[start];
			right = objs[start+1];
		}else{
			std::sort(std::begin(objs) + start,std::begin(objs) + end,comp);
			int mid = start + span/2;
			left = std::make_shared<BVH>(objs,start,mid,(axis+1)%3);
            right = std::make_shared<BVH>(objs,mid,end,(axis+1)%3);
		}

		bbox = AABB(left->getBox(), right->getBox());
	}
    

    ObjectInfo checkObject(Ray& ray) override{
        //hit nothing, return nothing
        if (!bbox.hit(ray))
            return ObjectInfo();

        ObjectInfo leftInfo = left->checkObject(ray);
        ObjectInfo rightInfo = right->checkObject(ray);

        if(leftInfo.isHit && !rightInfo.isHit) return leftInfo;
        else if(!leftInfo.isHit && rightInfo.isHit) return rightInfo;
        else if(!leftInfo.isHit && !rightInfo.isHit) return ObjectInfo();
        else{
            if(leftInfo.distance < rightInfo.distance) return leftInfo;
            else return rightInfo;
        }
    }

	static bool box_compare(
        const shared_ptr<Object> a, const shared_ptr<Object> b, int axis_index
    ) {
        auto a_axis_interval = a->getBox().getAxis(axis_index);
        auto b_axis_interval = b->getBox().getAxis(axis_index);
        return a_axis_interval.min < b_axis_interval.min;
    }

    static bool box_x_compare (const shared_ptr<Object> a, const shared_ptr<Object> b) {
        return box_compare(a, b, 0);
    }

    static bool box_y_compare (const shared_ptr<Object> a, const shared_ptr<Object> b) {
        return box_compare(a, b, 1);
    }

    static bool box_z_compare (const shared_ptr<Object> a, const shared_ptr<Object> b) {
        return box_compare(a, b, 2);
    }
	AABB getBox() override {return bbox;}
    
};


#endif
