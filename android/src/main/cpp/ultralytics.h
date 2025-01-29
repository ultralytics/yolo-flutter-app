//
// Created by Sergio Sánchez on 23/11/23.
//

#ifndef ANDROID_ULTRALYTICS_H
#define ANDROID_ULTRALYTICS_H

#include <algorithm>

#include "rect.h"

struct DetectedObject {
    Rect_<float> rect;
    int index;
    float confidence;
};

#endif //ANDROID_ULTRALYTICS_H
