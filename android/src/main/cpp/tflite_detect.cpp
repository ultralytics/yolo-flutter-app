#include <jni.h>
#include "ultralytics.h"

static void qsort_descent_inplace(std::vector<DetectedObject> &objects, int left, int right) {
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

static void qsort_descent_inplace(std::vector<DetectedObject> &objects) {
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static float intersection_area(const DetectedObject &a, const DetectedObject &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void nms_sorted_bboxes(const std::vector<DetectedObject> &objects, std::vector<int> &picked,
                              float nms_threshold) {
    picked.clear();

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = objects[i].rect.width * objects[i].rect.height;
    }

    for (int i = 0; i < n; i++) {
        const DetectedObject &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const DetectedObject &b = objects[picked[j]];

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
Java_com_ultralytics_ultralytics_1yolo_predict_detect_TfliteDetector_postprocess(JNIEnv *env,
                                                                                 jobject thiz,
                                                                                 jobjectArray recognitions,
                                                                                 jint w, jint h,
                                                                                 jfloat confidence_threshold,
                                                                                 jfloat iou_threshold,
                                                                                 jint num_items_threshold,
                                                                                 jint num_classes) {
    std::vector<DetectedObject> proposals;
    std::vector<DetectedObject> objects;

    // Initialize the C++ vector
    std::vector<std::vector<float>> vec(h, std::vector<float>(w, 0.0f));
    for (int i = 0; i < h; ++i) {
        jfloatArray row = (jfloatArray) env->GetObjectArrayElement(recognitions, i);
        jfloat *rowData = env->GetFloatArrayElements(row, JNI_FALSE);

        for (int j = 0; j < w; ++j) {
            vec[i][j] = rowData[j];
        }

        // Release the local references
        env->ReleaseFloatArrayElements(row, rowData, JNI_ABORT);
        env->DeleteLocalRef(row);
    }

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

            DetectedObject obj;
            obj.rect.x = dx;
            obj.rect.y = dy;
            obj.rect.width = dw;
            obj.rect.height = dh;
            obj.index = class_index;
            obj.confidence = class_score;

            proposals.push_back(obj);
        }
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, iou_threshold);

    int count = (int) std::min((float) picked.size(), (float) num_items_threshold);

    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        float x0 = std::max(0.f, objects[i].rect.x - objects[i].rect.width / 2);
        float y0 = std::max(0.f, objects[i].rect.y - objects[i].rect.height / 2);
        float x1 = std::min(1.f, objects[i].rect.x + objects[i].rect.width / 2);
        float y1 = std::min(1.f, objects[i].rect.y + objects[i].rect.height / 2);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = (x1 - x0);
        objects[i].rect.height = (y1 - y0);
    }

    //return 2-dimension array [detected_box][6(x, y, width, height, conf, class)]
    jobjectArray objArray;
    jclass floatArray = env->FindClass("[F");
    if (floatArray == NULL)
        return NULL;
    int size = objects.size();
    objArray = env->NewObjectArray(size, floatArray, NULL);
    if (objArray == NULL)
        return NULL;
    for (int i = 0; i < objects.size(); i++) {
        int index = objects[i].index;
        float x = objects[i].rect.x;
        float y = objects[i].rect.y;
        float width = objects[i].rect.width;
        float height = objects[i].rect.height;
        float confidence = objects[i].confidence;

        float boxres[6] = {x, y, width, height, confidence, (float) index};
        jfloatArray iarr = env->NewFloatArray((jsize) 6);
        if (iarr == NULL)
            return NULL;
        env->SetFloatArrayRegion(iarr, 0, 6, boxres);
        env->SetObjectArrayElement(objArray, i, iarr);
        env->DeleteLocalRef(iarr);
    }
    return objArray;
}