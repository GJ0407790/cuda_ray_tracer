/**
 * @file struct.hpp
 * @author Jin (jinj2@illinois.edu)
 * @brief 
 * @version 0.1
 * @date 2024-11-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef RAY_H
#define RAY_H

#include "libpng.h"
#include "vec3.hpp"
#include "interval.hpp"

using std::pow;

/*r,g,b from 0 to 1*/
class RGB{
public:
	double r = 0.0;
	double g = 0.0;
	double b = 0.0;
    RGB(): r(0.0),g(0.0),b(0.0){}
	RGB(double r,double g,double b): r(r),g(g),b(b){}
    bool operator==(const RGB& other) const {
		return (r == other.r && g == other.g && b == other.b);
	}
    RGB operator-(const RGB& other) const {
		return RGB(r-other.r,g-other.g,b-other.b);
	}
    RGB operator*(const RGB& other) const {
		return RGB(r*other.r,g*other.g,b*other.b);
	}
};

/*r,g,b from 0 to 1*/
class RGBA{
public:
	double r = 0.0;
	double g = 0.0;
	double b = 0.0;
	double a = 0.0;
	RGBA(): r(0.0),g(0.0),b(0.0),a(0.0){}
	RGBA(double r,double g,double b,double a): r(r),g(g),b(b),a(a){}
	RGBA operator+(const RGBA& other) const {
		return RGBA(r+other.r,g+other.g,b+other.b,a+other.a);
	}
    friend RGBA operator*(RGB rgb,const RGBA& other){
		return RGBA(rgb.r*other.r,rgb.g*other.g,rgb.b*other.b,other.a);
	}
	RGBA mean(int aa){
		return RGBA(r/aa,g/aa,b/aa,a/aa);
	}
};


/*u,v*/
typedef struct{
	double u = 0;
	double v = 0;
}Texcoord;

/**
 * @brief Class Ray, consists of a eye and direction
 */
class Ray{
public:
	point3 eye;
	vec3 dir;
	int bounce;
    Ray() : bounce(0) {}
	Ray(double x,double y);
	Ray(point3 eye,vec3 dir,int bounce): eye(eye), dir(dir.normalize()),bounce(bounce){}
	
};

class AABB{
public:
    Interval x,y,z;

    AABB() {}

    AABB(const Interval& x,const Interval& y,const Interval& z):
        x(x),y(y),z(z) {}
    
    AABB(const point3& a,const point3& b) {
        x = (a.x <= b.x) ? Interval(a.x, b.x) : Interval(b.x, a.x);
        y = (a.y <= b.y) ? Interval(a.y, b.y) : Interval(b.y, a.y);
        z = (a.z <= b.z) ? Interval(a.z, b.z) : Interval(b.z, a.z);
    }

    AABB(const point3& a,const point3& b,const point3& c) {
        float x_min,x_max,y_min,y_max,z_min,z_max;
        //obtain the min and max value for the three coordinates
        x_min = std::min({a.x,b.x,c.x});x_max = std::max({a.x,b.x,c.x});
        y_min = std::min({a.y,b.y,c.y});y_max = std::max({a.y,b.y,c.y});
        z_min = std::min({a.z,b.z,c.z});z_max = std::max({a.z,b.z,c.z});
        x = Interval(x_min,x_max);y = Interval(y_min,y_max);z = Interval(z_min,z_max);
        //expand if the interval is too small
        if(x.size() < 0.01) x = x.expand(0.01);
        if(y.size() < 0.01) y = y.expand(0.01);
        if(z.size() < 0.01) z = z.expand(0.01);
    }

    AABB(const AABB& a, const AABB& b) {
        x = Interval(std::min(a.x.min,b.x.min),std::max(a.x.max,b.x.max));
        y = Interval(std::min(a.y.min,b.y.min),std::max(a.y.max,b.y.max));
        z = Interval(std::min(a.z.min,b.z.min),std::max(a.z.max,b.z.max));
    }

    const Interval& getAxis(int n) const {
        if (n == 0) return x;
        if (n == 1) return y;
        return z;
    }

    int longestAxis() const {
        if (x.size() > y.size())
            return x.size() > z.size() ? 0 : 2;
        else
            return y.size() > z.size() ? 1 : 2;
    }


    bool hit(const Ray& r) const {
        const point3& ray_eye = r.eye;
        const vec3&   ray_dir  = r.dir;

        double t_min = double(-INFINITY);
        double t_max = double(INFINITY);

        for (int axis = 0; axis < 3; axis++) {
            const Interval& ax = getAxis(axis);
            if (ax.min > ax.max) return false;
            //const double adinv = (ray_dir[axis] != 0) ? (1.0 / ray_dir[axis]) : INFINITY;
            const double adinv = 1.0 / ray_dir[axis];

            double t0 = (ax.min - ray_eye[axis]) * adinv;
            double t1 = (ax.max - ray_eye[axis]) * adinv;

            if (t0 > t1) std::swap(t0, t1);

            t_min = std::max(t_min, t0);
            t_max = std::min(t_max, t1);

            if (t_max <= t_min)
                return false;
        }
        return true;
    }
};

#endif
