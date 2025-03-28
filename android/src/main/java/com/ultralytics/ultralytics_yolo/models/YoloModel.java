// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.ultralytics_yolo.models;

import java.util.ArrayList;

public abstract class YoloModel {
    public final ArrayList<String> labels = new ArrayList<>();
    public String task;
    public String format;
}