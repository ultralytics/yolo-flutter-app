package com.ultralytics.ultralytics_yolo.predict.classify;

public class ClassificationResult {
    public final String label;
    public final float confidence;
    public final int index;

    public ClassificationResult(String label, int index, float confidence) {
        this.label = label;
        this.index = index;
        this.confidence = confidence;
    }
}
