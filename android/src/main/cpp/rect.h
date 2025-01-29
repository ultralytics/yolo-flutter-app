#ifndef RECT_H
#define RECT_H

#include <algorithm>
#include <iostream>

template <typename _Tp>
class Rect_ {
public:
    // Type alias for value type
    typedef _Tp value_type;

    // Public members
    _Tp x, y, width, height;

    // Default constructor
    Rect_() : x(0), y(0), width(0), height(0) {}

    // Constructor with values
    Rect_(_Tp _x, _Tp _y, _Tp _width, _Tp _height)
        : x(_x), y(_y), width(_width), height(_height) {}

    // Intersection operator (&) to compute the intersection of two Rect_ objects
    Rect_<_Tp> operator&(const Rect_<_Tp>& rhs) const {
        _Tp x1 = std::max(x, rhs.x);
        _Tp y1 = std::max(y, rhs.y);
        _Tp x2 = std::min(x + width, rhs.x + rhs.width);
        _Tp y2 = std::min(y + height, rhs.y + rhs.height);

        // If there's no intersection, return an empty Rect_
        if (x2 <= x1 || y2 <= y1) {
            return Rect_<_Tp>();
        }
        return Rect_<_Tp>(x1, y1, x2 - x1, y2 - y1);
    }

    // Method to calculate area of the rectangle
    _Tp area() const {
        return width * height;
    }

    // Method to check if a point is inside the rectangle
    bool contains(_Tp px, _Tp py) const {
        return (px >= x && px <= x + width && py >= y && py <= y + height);
    }

    // Method to check if two rectangles intersect
    bool intersects(const Rect_<_Tp>& rhs) const {
        return (std::max(x, rhs.x) < std::min(x + width, rhs.x + rhs.width) &&
                std::max(y, rhs.y) < std::min(y + height, rhs.y + rhs.height));
    }

    // Method to display the rectangle (for debugging)
    void display() const {
        std::cout << "Rect_(" << x << ", " << y << ", " << width << ", " << height << ")\n";
    }
};

#endif // RECT_H
