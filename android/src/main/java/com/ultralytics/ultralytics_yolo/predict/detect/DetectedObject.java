package com.ultralytics.ultralytics_yolo.predict.detect;

import android.graphics.RectF;

import androidx.annotation.Keep;

public class DetectedObject {
    public final Float confidence;
    public final RectF boundingBox;
    public final int index;
    public final String label;

    public DetectedObject(final Float confidence, final RectF boundingBox, final int index, final String label) {
        this.confidence = confidence;
        this.boundingBox = boundingBox;
        this.index = index;
        this.label = label;
    }

    @Keep
    public Float getConfidence() {
        return confidence;
    }

    @Keep
    public RectF getBoundingBox() {
        return new RectF(boundingBox);
    }

    @Keep
    public int getIndex() {
        return index;
    }

    @Keep
    public String getLabel() {
        return label;
    }
}

