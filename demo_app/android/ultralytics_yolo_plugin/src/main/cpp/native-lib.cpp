// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// app/src/main/cpp/native-lib.cpp

#include <jni.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdlib>

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
static void nms_sorted_bboxes(const std::vector<DetectedObject>& objects, std::vector<int>& picked, float nms_threshold) {
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
        if (keep)
            picked.push_back(i);
    }
}

extern "C"
JNIEXPORT jobjectArray JNICALL
Java_com_ultralytics_yolo_ObjectDetector_postprocess(
        JNIEnv *env,
        jobject thiz,
        jobjectArray recognitions,
        jint w, jint h,
        jfloat confidence_threshold,
        jfloat iou_threshold,
        jint num_items_threshold,
        jint num_classes) {

    // Convert 2D array to C++ vector
    std::vector<std::vector<float>> vec;
    vec.resize(h, std::vector<float>(w, 0.0f));
    for (int i = 0; i < h; ++i) {
        jfloatArray row = (jfloatArray) env->GetObjectArrayElement(recognitions, i);
        jfloat* rowData = env->GetFloatArrayElements(row, JNI_FALSE);
        for (int j = 0; j < w; ++j) {
            vec[i][j] = rowData[j];
        }
        env->ReleaseFloatArrayElements(row, rowData, JNI_ABORT);
        env->DeleteLocalRef(row);
    }

    // Extract box candidates (proposals)
    std::vector<DetectedObject> proposals;
    for (int i = 0; i < w; ++i) {
        int class_index = 0;
        float class_score = -FLT_MAX;
        // Get each class score (assuming class scores start from the 4th index)
        for (int c = 0; c < num_classes; c++) {
            float score = vec[c + 4][i];
            if (score > class_score) {
                class_score = score;
                class_index = c;
            }
        }
        // Only add to candidates if score exceeds threshold
        if (class_score > confidence_threshold) {
            // Get center coordinates and width/height, convert to top-left coordinates
            float cx = vec[0][i];
            float cy = vec[1][i];
            float w_box = vec[2][i];
            float h_box = vec[3][i];

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

    // Sort by score
    qsort_descent_inplace(proposals);

    // Apply Non-Maximum Suppression (NMS)
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, iou_threshold);

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