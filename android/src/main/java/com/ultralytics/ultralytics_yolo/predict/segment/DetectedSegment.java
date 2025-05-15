// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.ultralytics_yolo.predict.segment;

import android.graphics.RectF;

import androidx.annotation.Keep;

import java.util.List;

public class DetectedSegment {
    public final Float confidence;
    public final RectF boundingBox;
    public final int index;
    public final String label;
    public final List<Float> mask; // Flattened mask data

    public DetectedSegment(final Float confidence, final RectF boundingBox, final int index, final String label, final List<Float> mask) {
        this.confidence = confidence;
        this.boundingBox = boundingBox;
        this.index = index;
        this.label = label;
        this.mask = mask;
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

    @Keep
    public List<Float> getMask() {
        return mask;
    }
}