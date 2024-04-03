package com.ultralytics.ultralytics_yolo.predict.detect;

import android.content.Context;

import androidx.annotation.Keep;

import com.ultralytics.ultralytics_yolo.predict.Predictor;

public abstract class Detector extends Predictor {
    protected Detector(Context context) {
        super(context);
    }

    public abstract void setObjectDetectionResultCallback(ObjectDetectionResultCallback callback);

    public abstract void setIouThreshold(float iou);

    public abstract void setNumItemsThreshold(int numItems);

    public interface ObjectDetectionResultCallback {
        @Keep()
        void onResult(float[][] detections);
    }
}
