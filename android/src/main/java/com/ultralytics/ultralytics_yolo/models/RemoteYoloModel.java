package com.ultralytics.ultralytics_yolo.models;

public class RemoteYoloModel extends YoloModel {
    public final String modelUrl;
    public final String labelsUrl;

    public RemoteYoloModel(String modelUrl, String task) {
        super.task = task;
        this.modelUrl = modelUrl;
        labelsUrl = null;
    }

    public RemoteYoloModel(String modelUrl, String labelsUrl, String task) {
        super.task = task;
        this.modelUrl = modelUrl;
        this.labelsUrl = labelsUrl;
    }
}
