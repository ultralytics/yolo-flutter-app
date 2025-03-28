// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.ultralytics_yolo.models;

public class LocalYoloModel extends YoloModel {
    public final String modelPath;
    public final String metadataPath;

    public LocalYoloModel(String task, String format, String modelPath, String metadataPath) {
        super.task = task;
        super.format = format;
        this.modelPath = modelPath;
        this.metadataPath = metadataPath;
    }
}
