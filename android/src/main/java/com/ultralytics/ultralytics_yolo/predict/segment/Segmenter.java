// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.ultralytics_yolo.predict.segment;

import android.content.Context;

import androidx.annotation.Keep;

import com.ultralytics.ultralytics_yolo.predict.Predictor;

public abstract class Segmenter extends Predictor {
    protected Segmenter(Context context) {
        super(context);
    }

    public abstract void setSegmentationResultCallback(SegmentationResultCallback callback);

    public abstract void setIouThreshold(float iou);

    public abstract void setNumItemsThreshold(int numItems);

    public interface SegmentationResultCallback {
        @Keep()
        void onResult(Object[] segmentationResults);
    }
}