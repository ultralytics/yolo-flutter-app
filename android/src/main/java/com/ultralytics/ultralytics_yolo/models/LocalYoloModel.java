package com.ultralytics.ultralytics_yolo.models;

public class LocalYoloModel extends YoloModel {
    public final String modelPath;
    public final String metadataPath;

    public LocalYoloModel(String task, String format, String modelPath, String metadataPath, boolean isLive) {
        super.task = task;
        super.format = format;
        super.isLive = isLive;
        this.modelPath = modelPath;
        this.metadataPath = metadataPath;
    }
}
