// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
// Created by Sergio Sánchez on 23/11/23.
//

#ifndef ANDROID_ULTRALYTICS_H
#define ANDROID_ULTRALYTICS_H

struct DetectedObject {
    cv::Rect_<float> rect;
    int index;
    float confidence;
};

#endif //ANDROID_ULTRALYTICS_H
