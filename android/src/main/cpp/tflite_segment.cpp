// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include <jni.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
// #include <opencv2/core/core.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
#include <android/log.h> // For logging
#include <cstring>
#include "ultralytics.h" // For memcpy
#include <opencv2/opencv.hpp>


#define LOG_TAG "TfliteSegmenterCpp"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// struct DetectedSegmentObject {
//     cv::Rect_<float> rect;
//     int class_id;
//     float confidence;
//     std::vector<float> mask; // Mask coefficients (32)
// };
struct DetectedSegmentObject {
    cv::Rect_<float> rect;
    int index;
    float confidence;            // Detection confidence score
    std::vector<float> mask_coeff; // Mask coefficients from raw_detections, size = mask_channels
};

static void qsort_descent_inplace(std::vector<DetectedSegmentObject> &objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].confidence;

    while (i <= j) {
        while (objects[i].confidence > p)
            i++;

        while (objects[j].confidence < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

static float intersection_area(const DetectedSegmentObject &a, const DetectedSegmentObject &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<DetectedSegmentObject> &objects) {
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}



std::vector<std::vector<cv::Point>> get_polygons(const std::vector<std::vector<float>>& mask) {
    if (mask.empty() || mask[0].empty()) {
        LOGD("Warning: Input mask is empty.");
        return {};
    }

    int rows = mask.size();
    int cols = mask[0].size();
    cv::Mat cv_mask(rows, cols, CV_8U);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cv_mask.at<uchar>(i, j) = static_cast<uchar>(mask[i][j] > 0.5f ? 255 : 0); 
           // Thresholding
        }
    }

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cv_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> polygons;
    for (const auto& contour : contours) {
        if (!contour.empty()) {
            polygons.push_back(contour);
        }
    }
    return polygons;
}

std::vector<cv::Point> get_outer_contour_convex(const std::vector<std::vector<cv::Point>>& polygons) {
    std::vector<cv::Point> all_points;
    for (const auto& poly : polygons) {
        all_points.insert(all_points.end(), poly.begin(), poly.end());
    }

    std::vector<cv::Point> hull;
    if (!all_points.empty()) {
        cv::convexHull(all_points, hull);
    }
    return hull;
}

std::vector<cv::Point> get_outer_contour_findcontours(const std::vector<std::vector<cv::Point>>& polygons, int image_width, int image_height) {
    if (polygons.empty()) {
        return {};
    }

    // 1. Create a blank black image
    cv::Mat mask(image_height, image_width, CV_8U, cv::Scalar(0));

    // 2. Draw all the input polygons onto the mask with white color
    for (const auto& poly : polygons) {
        std::vector<std::vector<cv::Point>> contour = {poly}; // findContours expects a vector of vectors
        cv::drawContours(mask, contour, 0, cv::Scalar(255), cv::FILLED);
    }

    // 3. Find the contours in the resulting binary image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 4. Find the contour with the largest area (assuming it's the outer one)
    double max_area = 0;
    int max_area_contour_index = -1;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_area_contour_index = i;
        }
    }

    // 5. Return the largest contour, or an empty vector if no contours were found
    if (max_area_contour_index >= 0) {
        return contours[max_area_contour_index];
    } else {
        return {};
    }
}





static void nms_sorted_bboxes(const std::vector<DetectedSegmentObject> &objects, std::vector<int> &picked,
    float nms_threshold) {
picked.clear();

const int n = objects.size();

std::vector<float> areas(n);
for (int i = 0; i < n; i++) {
areas[i] = objects[i].rect.width * objects[i].rect.height;
}

for (int i = 0; i < n; i++) {
const DetectedSegmentObject &a = objects[i];

int keep = 1;
for (int j = 0; j < (int) picked.size(); j++) {
const DetectedSegmentObject &b = objects[picked[j]];

// intersection over union
float inter_area = intersection_area(a, b);
float union_area = areas[i] + areas[picked[j]] - inter_area;
// float IoU = inter_area / union_area
if (inter_area / union_area > nms_threshold)
keep = 0;
}

if (keep)
picked.push_back(i);
}
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_ultralytics_ultralytics_1yolo_predict_segment_TfliteSegmenter_postprocess(
    JNIEnv *env,
    jobject thiz,
    jobjectArray raw_detections_obj, // [1, 37, 8400]
    jobjectArray mask_protos_obj,     // [1, 160, 160, 32]
    jint w, jint h,
    jfloat confidence_threshold,
    jfloat iou_threshold,
    jint num_items_threshold,
    jint num_classes, // Should be 1 for your case
    jint mask_channels, // Should be 32 for your case
    jint mask_shape0,
    jint mask_shape1,
    jint mask_shape2,
    jint mask_shape3

) {

    

    std::vector<DetectedSegmentObject> proposals;
    std::vector<DetectedSegmentObject> objects;

    jobjectArray detections_batch = (jobjectArray) env->GetObjectArrayElement(raw_detections_obj, 0);
if (detections_batch == nullptr) {
    LOGD("Error: detections_batch is null.");
    return nullptr;
}
    // Initialize the C++ vector
    std::vector<std::vector<float>> vec(h, std::vector<float>(w, 0.0f));
    for (int i = 0; i < h; ++i) {
        jfloatArray row = (jfloatArray) env->GetObjectArrayElement(detections_batch, i);
        jfloat *rowData = env->GetFloatArrayElements(row, JNI_FALSE);

        for (int j = 0; j < w; ++j) {
            vec[i][j] = rowData[j];
        }

        // Release the local references
        env->ReleaseFloatArrayElements(row, rowData, JNI_ABORT);
        env->DeleteLocalRef(row);
    }

    jobjectArray proto_160 = (jobjectArray) env->GetObjectArrayElement(mask_protos_obj, 0);
std::vector<std::vector<std::vector<float>>> protos(mask_shape3, std::vector<std::vector<float>>(mask_shape2, std::vector<float>(mask_shape1)));

for (int y = 0; y < mask_shape1; ++y) {
    jobjectArray proto_row = (jobjectArray) env->GetObjectArrayElement(proto_160, y);
    for (int x = 0; x < mask_shape2; ++x) {
        jfloatArray proto_pixel = (jfloatArray) env->GetObjectArrayElement(proto_row, x);
        jfloat* proto_vals = env->GetFloatArrayElements(proto_pixel, 0);
        for (int c = 0; c < mask_shape3; ++c) {
            protos[c][y][x] = proto_vals[c];
        }
        env->ReleaseFloatArrayElements(proto_pixel, proto_vals, 0);
        env->DeleteLocalRef(proto_pixel);
    }
    env->DeleteLocalRef(proto_row);
}
env->DeleteLocalRef(proto_160);

    // find boxes with score > threshold and class > threshold
    for (int i = 0; i < w; ++i) {
        // find class index with max class score
        int class_index = 0;
        float class_score = -FLT_MAX;

        // get scores for all class indexes of current box
        std::vector<float> classes(num_classes);
        for (int c = 0; c < num_classes; c++) {
            classes[c] = vec[c + 4][i];
        }

        // find class index with max class score
        for (int c = 0; c < num_classes; ++c) {
            if (classes[c] > class_score) {
                class_index = c;
                class_score = classes[c];
            }
        }

        // if class score is less than threshold, move to next box
        if (class_score > confidence_threshold) {
           
            float dx = vec[0][i];
            float dy = vec[1][i];
            float dw = vec[2][i];
            float dh = vec[3][i];

            DetectedSegmentObject obj;

            obj.mask_coeff.resize(mask_channels);
for (int c = 0; c < mask_channels; ++c) {
    obj.mask_coeff[c] = vec[c + 4 + num_classes][i];
}
            obj.rect.x = dx;
            obj.rect.y = dy;
            obj.rect.width = dw;
            obj.rect.height = dh;
            obj.index = class_index;
            obj.confidence = class_score;

            proposals.push_back(obj);
        }
    }

    int n = proposals.size();
    std::vector<std::vector<std::vector<cv::Point>>> all_polygons;

    // Prepare mask coefficients and prototype masks (your existing code)
    std::vector<std::vector<float>> coeff_mat(n, std::vector<float>(mask_shape3));
    std::vector<std::vector<float>> protos_mat(mask_shape3, std::vector<float>(mask_shape2 * mask_shape1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < mask_shape3; ++j) {
            coeff_mat[i][j] = proposals[i].mask_coeff[j];
        }
    }
    for (int c = 0; c < mask_shape3; ++c) {
        for (int y = 0; y < mask_shape1; ++y) {
            for (int x = 0; x < mask_shape2; ++x) {
                protos_mat[c][y * mask_shape2 + x] = protos[c][y][x];
            }
        }
    }

    // Multiply to get flat masks
    std::vector<std::vector<float>> mask_flat(n, std::vector<float>(mask_shape1 * mask_shape2, 0.0f));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < mask_shape1 * mask_shape2; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < mask_shape3; ++k) {
                sum += coeff_mat[i][k] * protos_mat[k][j];
            }
            mask_flat[i][j] = sum;
        }
    }

    std::vector<std::vector<std::vector<float>>> final_masks(n, std::vector<std::vector<float>>(mask_shape1, std::vector<float>(mask_shape2)));
    for (int i = 0; i < n; ++i) {
        for (int y = 0; y < mask_shape1; ++y) {
            for (int x = 0; x < mask_shape2; ++x) {
                final_masks[i][y][x] = mask_flat[i][y * mask_shape2 + x];
            }
        }
    }

    

    // Generate polygons for all initial proposals
    for (const auto& mask_2d : final_masks) {
        all_polygons.push_back(get_polygons(mask_2d));
    }
   
    // NMS and object filtering
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, iou_threshold);
    int count = (int) std::min((float) picked.size(), (float) num_items_threshold);
    objects.resize(count);
    std::vector<std::vector<std::vector<cv::Point>>> picked_polygons(count); // Polygons for the picked objects

    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];
       // picked_polygons[i] = all_polygons[picked[i]]; // Get corresponding polygons

      

        float x0 = std::max(0.f, objects[i].rect.x - objects[i].rect.width / 2);
        float y0 = std::max(0.f, objects[i].rect.y - objects[i].rect.height / 2);
        float x1 = std::min(1.f, objects[i].rect.x + objects[i].rect.width / 2);
        float y1 = std::min(1.f, objects[i].rect.y + objects[i].rect.height / 2);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = (x1 - x0);
        objects[i].rect.height = (y1 - y0);

        cv::Rect pixel_rect(
            objects[i].rect.x * mask_shape1,
            objects[i].rect.y * mask_shape2,
            objects[i].rect.width * mask_shape1,
            objects[i].rect.height * mask_shape2
        );
        
        std::vector<std::vector<cv::Point>> filtered_polygons;
        for (const auto& polygon_group : all_polygons) {
            for (const auto& polygon : polygon_group) {
                std::vector<cv::Point> filtered_points;
                for (const auto& point : polygon) {
                    if (pixel_rect.contains(point)) {
                        filtered_points.push_back(point);
                    }
                }
                if (!filtered_points.empty()) {
                    filtered_polygons.push_back(filtered_points);
                }
            }
        }
       
        
        //picked_polygons[i] =  get_outer_contour_convex(filtered_polygons);
        picked_polygons[i].push_back(get_outer_contour_findcontours(filtered_polygons,w,h));
    }

    // Create the array of result objects
    jclass objectClass = env->FindClass("java/lang/Object"); // Use a generic Object class
    jobjectArray resultObjArray = env->NewObjectArray(count, objectClass, nullptr);
    if (resultObjArray == nullptr) return nullptr;

    // Prepare classes and methods for creating Maps and Lists
    jclass hashMapClass = env->FindClass("java/util/HashMap");
    jmethodID hashMapConstructor = env->GetMethodID(hashMapClass, "<init>", "()V");
    jmethodID hashMapPut = env->GetMethodID(hashMapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");

    jclass arrayListClass = env->FindClass("java/util/ArrayList");
    jmethodID arrayListConstructor = env->GetMethodID(arrayListClass, "<init>", "()V");
    jmethodID arrayListAdd = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");

    jclass floatClazz = env->FindClass("java/lang/Float");
    jmethodID floatConstructor = env->GetMethodID(floatClazz, "<init>", "(F)V");

    jclass integerClazz = env->FindClass("java/lang/Integer");
    jmethodID integerConstructor = env->GetMethodID(integerClazz, "<init>", "(I)V");

    // Populate the result array
    for (int i = 0; i < count; ++i) {
        jobject resultMap = env->NewObject(hashMapClass, hashMapConstructor);

        // Add bounding box information to the map
        env->CallObjectMethod(resultMap, hashMapPut, env->NewStringUTF("x"), env->NewObject(floatClazz, floatConstructor, objects[i].rect.x));
        env->CallObjectMethod(resultMap, hashMapPut, env->NewStringUTF("y"), env->NewObject(floatClazz, floatConstructor, objects[i].rect.y));
        env->CallObjectMethod(resultMap, hashMapPut, env->NewStringUTF("width"), env->NewObject(floatClazz, floatConstructor, objects[i].rect.width));
        env->CallObjectMethod(resultMap, hashMapPut, env->NewStringUTF("height"), env->NewObject(floatClazz, floatConstructor, objects[i].rect.height));
        env->CallObjectMethod(resultMap, hashMapPut, env->NewStringUTF("confidence"), env->NewObject(floatClazz, floatConstructor, objects[i].confidence));
        env->CallObjectMethod(resultMap, hashMapPut, env->NewStringUTF("class"), env->NewObject(integerClazz, integerConstructor, objects[i].index));

        // Add polygon information to the map
        jobject polygonsList = env->NewObject(arrayListClass, arrayListConstructor);
        for (const auto& polygon : picked_polygons[i]) {
            jobject pointList = env->NewObject(arrayListClass, arrayListConstructor);
            for (const auto& point : polygon) {
                jobject pointMap = env->NewObject(hashMapClass, hashMapConstructor);
                env->CallObjectMethod(pointMap, hashMapPut, env->NewStringUTF("x"), env->NewObject(integerClazz, integerConstructor, point.x));
                env->CallObjectMethod(pointMap, hashMapPut, env->NewStringUTF("y"), env->NewObject(integerClazz, integerConstructor, point.y));
                env->CallBooleanMethod(pointList, arrayListAdd, pointMap);
                env->DeleteLocalRef(pointMap);
            }
            env->CallBooleanMethod(polygonsList, arrayListAdd, pointList);
            env->DeleteLocalRef(pointList);
        }
        env->CallObjectMethod(resultMap, hashMapPut, env->NewStringUTF("polygons"), polygonsList);
        env->DeleteLocalRef(polygonsList);

        env->SetObjectArrayElement(resultObjArray, i, resultMap);
        env->DeleteLocalRef(resultMap);
    }
    jsize length = env->GetArrayLength(resultObjArray);
    return resultObjArray;
}