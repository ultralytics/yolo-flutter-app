package com.ultralytics.ultralytics_yolo.models;

public class RemoteYoloModel extends YoloModel {
    public final String modelUrl;
    public final String labelsUrl;

    public RemoteYoloModel(String modelUrl, String task, boolean isLive) {
        super.task = task;
        super.isLive = isLive;
        this.modelUrl = modelUrl;
        labelsUrl = null;
    }

    public RemoteYoloModel(String modelUrl, String labelsUrl, String task, boolean isLive) {
        super.task = task;
        super.isLive = isLive;
        this.modelUrl = modelUrl;
        this.labelsUrl = labelsUrl;
    }
}
