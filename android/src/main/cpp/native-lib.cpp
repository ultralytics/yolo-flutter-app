// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// app/src/main/cpp/native-lib.cpp

#include <jni.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdlib>
#include "depth-colorizer.h"

// Custom rectangle structure
struct Rect {
    float x;
    float y;
    float width;
    float height;
};

// Detected object structure
struct DetectedObject {
    Rect rect;
    int index;
    float confidence;
};

// Quicksort for descending order
static void qsort_descent_inplace(std::vector<DetectedObject>& objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].confidence;
    while (i <= j) {
        while (objects[i].confidence > p) i++;
        while (objects[j].confidence < p) j--;
        if (i <= j) {
            std::swap(objects[i], objects[j]);
            i++; j--;
        }
    }
    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}

static void qsort_descent_inplace(std::vector<DetectedObject>& objects) {
    if (!objects.empty())
        qsort_descent_inplace(objects, 0, objects.size() - 1);
}

// Calculate intersection area (common area of two rectangles)
static float intersection_area(const DetectedObject &a, const DetectedObject &b) {
    float ax1 = a.rect.x;
    float ay1 = a.rect.y;
    float ax2 = a.rect.x + a.rect.width;
    float ay2 = a.rect.y + a.rect.height;
    float bx1 = b.rect.x;
    float by1 = b.rect.y;
    float bx2 = b.rect.x + b.rect.width;
    float by2 = b.rect.y + b.rect.height;

    float interX1 = std::max(ax1, bx1);
    float interY1 = std::max(ay1, by1);
    float interX2 = std::min(ax2, bx2);
    float interY2 = std::min(ay2, by2);

    float interWidth = std::max(0.0f, interX2 - interX1);
    float interHeight = std::max(0.0f, interY2 - interY1);

    return interWidth * interHeight;
}

// Non-Maximum Suppression (NMS) implementation (for already sorted proposals)
static void nms_sorted_bboxes(const std::vector<DetectedObject>& objects, std::vector<int>& picked, float nms_threshold, int max_picked) {
    picked.clear();
    int n = objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = objects[i].rect.width * objects[i].rect.height;
    }
    for (int i = 0; i < n; i++) {
        const DetectedObject &a = objects[i];
        bool keep = true;
        for (int j = 0; j < (int)picked.size(); j++) {
            const DetectedObject &b = objects[picked[j]];
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (union_area > 0 && (inter_area / union_area > nms_threshold)) {
                keep = false;
                break;
            }
        }
        if (keep) {
            picked.push_back(i);
            if (max_picked > 0 && (int)picked.size() >= max_picked) break;
        }
    }
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_ultralytics_yolo_ObjectDetector_postprocess(
        JNIEnv *env,
        jobject thiz,
        jfloatArray recognitions,
        jint w, jint h,
        jfloat confidence_threshold,
        jfloat iou_threshold,
        jint num_items_threshold,
        jint num_classes) {

    // Read the flat [h x w] prediction tensor directly - one pin, no per-row marshaling or nested copies.
    jfloat *data = env->GetFloatArrayElements(recognitions, nullptr);
    if (data == nullptr) return NULL;

    // Extract box candidates (proposals); confidence first so box reads are skipped for rejected anchors
    std::vector<DetectedObject> proposals;
    for (int i = 0; i < w; ++i) {
        int class_index = 0;
        float class_score = -FLT_MAX;
        // Get each class score (class scores start at row 4)
        for (int c = 0; c < num_classes; c++) {
            float score = data[(c + 4) * w + i];
            if (score > class_score) {
                class_score = score;
                class_index = c;
            }
        }
        // Only add to candidates if score exceeds threshold
        if (class_score > confidence_threshold) {
            // Get center coordinates and width/height, convert to top-left coordinates
            float cx = data[i];
            float cy = data[w + i];
            float w_box = data[2 * w + i];
            float h_box = data[3 * w + i];

            DetectedObject obj;
            obj.rect.x = cx - w_box / 2;
            obj.rect.y = cy - h_box / 2;
            obj.rect.width = w_box;
            obj.rect.height = h_box;
            obj.index = class_index;
            obj.confidence = class_score;

            proposals.push_back(obj);
        }
    }
    env->ReleaseFloatArrayElements(recognitions, data, JNI_ABORT);

    // Sort by score
    qsort_descent_inplace(proposals);

    // Apply Non-Maximum Suppression (NMS)
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, iou_threshold, num_items_threshold);

    int count = std::min((int)picked.size(), (int)num_items_threshold);
    std::vector<DetectedObject> objects(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];
        // No additional conversion needed here for the Java version
    }

    // Return results as 2D array (each element: [x, y, width, height, confidence, class_index])
    jclass floatArrayCls = env->FindClass("[F");
    if (floatArrayCls == NULL) return NULL;
    jobjectArray objArray = env->NewObjectArray(objects.size(), floatArrayCls, NULL);
    if (objArray == NULL) return NULL;
    for (int i = 0; i < objects.size(); i++) {
        float box[6] = {
                objects[i].rect.x,
                objects[i].rect.y,
                objects[i].rect.width,
                objects[i].rect.height,
                objects[i].confidence,
                static_cast<float>(objects[i].index)
        };
        jfloatArray iarr = env->NewFloatArray(6);
        if (iarr == NULL) return NULL;
        env->SetFloatArrayRegion(iarr, 0, 6, box);
        env->SetObjectArrayElement(objArray, i, iarr);
        env->DeleteLocalRef(iarr);
    }
    return objArray;
}

extern "C"
JNIEXPORT jfloatArray JNICALL
Java_com_ultralytics_yolo_DepthEstimator_colorizeDepth(
        JNIEnv *env,
        jobject thiz,
        jfloatArray output,
        jint depth_width,
        jint left,
        jint top,
        jint width,
        jint height,
        jintArray color_pixels,
        jintArray colors) {
    if (depth_width <= 0 || left < 0 || top < 0 || width <= 0 || height <= 0 ||
        left + width > depth_width ||
        static_cast<jlong>(top + height) * depth_width > env->GetArrayLength(output) ||
        static_cast<jlong>(width) * height > env->GetArrayLength(color_pixels) ||
        env->GetArrayLength(colors) < 256) {
        return nullptr;
    }

    jfloat *depth = env->GetFloatArrayElements(output, nullptr);
    jint *pixels = env->GetIntArrayElements(color_pixels, nullptr);
    jint *color_table = env->GetIntArrayElements(colors, nullptr);
    if (depth == nullptr || pixels == nullptr || color_table == nullptr) {
        if (depth != nullptr) env->ReleaseFloatArrayElements(output, depth, JNI_ABORT);
        if (pixels != nullptr) env->ReleaseIntArrayElements(color_pixels, pixels, 0);
        if (color_table != nullptr) env->ReleaseIntArrayElements(colors, color_table, JNI_ABORT);
        return nullptr;
    }

    DepthRange range;
    const bool valid = colorize_depth(
            depth, depth_width, left, top, width, height, pixels, color_table, range);

    env->ReleaseFloatArrayElements(output, depth, JNI_ABORT);
    env->ReleaseIntArrayElements(color_pixels, pixels, 0);
    env->ReleaseIntArrayElements(colors, color_table, JNI_ABORT);

    if (!valid) return nullptr;
    jfloat result_range[2] = {range.min, range.max};
    jfloatArray result = env->NewFloatArray(2);
    if (result != nullptr) env->SetFloatArrayRegion(result, 0, 2, result_range);
    return result;
}
