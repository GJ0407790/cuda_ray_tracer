/**
 * @file interval.hpp
 * @author Jin (jinj2@illinois.edu)
 * @brief 
 * @version 0.1
 * @date 2024-11-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#ifndef BVH_H
#define BVH_H

#include "vec3.hpp"
#include <cmath>
#include <algorithm>
#include <memory>
#include <vector>
using std::shared_ptr;

class Interval {
public:
    double min,max;

    Interval() : min(+INFINITY),max(-INFINITY) {}

    Interval(double min,double max): 
            min(min),max(max) {}

    double size() const {return max - min;}

    /**
     * @brief Returns if the value is within, border value returns true.
     * @param x The value to be checked.
     * @return true 
     * @return false 
     */
    bool contains(double x) const {return min <= x && x <= max;}

    /**
     * @brief Returns if the value is within, border value returns false.
     * @param x The value to be checked.
     * @return true 
     * @return false 
     */
    bool surrounds(double x) const {return min <= x && x <= max;}

    /**
     * @brief Expand the interval by delta on both ends.
     * @param delta The value to expand on both ends.
     * @return Interval The expanded interval.
     * @details This function is used for the bounding box creation for
     *          triangles
     */
    Interval expand(double delta) const {
        return Interval(min - delta,max + delta);
    }
};

#endif
