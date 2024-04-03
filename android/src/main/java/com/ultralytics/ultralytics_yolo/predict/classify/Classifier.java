package com.ultralytics.ultralytics_yolo.predict.classify;

import android.content.Context;

import androidx.annotation.Keep;

import com.ultralytics.ultralytics_yolo.predict.Predictor;

import java.util.List;

public abstract class Classifier extends Predictor {
    protected Classifier(Context context) {
        super(context);
    }

    public abstract void setClassificationResultCallback(ClassificationResultCallback callback);

    public abstract void setNumItemsThreshold(int numItems);

    public interface ClassificationResultCallback {
        @Keep()
        void onResult(List<ClassificationResult> classificationResults);
    }
}
