package com.ultralytics.ultralytics_yolo;

import android.os.Handler;
import android.os.Looper;

import java.util.List;
import java.util.Map;

import io.flutter.plugin.common.EventChannel;

class ResultStreamHandler implements EventChannel.StreamHandler {
    final private Handler handler = new Handler(Looper.getMainLooper());
    private EventChannel.EventSink eventSink;

    @Override
    public void onListen(Object arguments, EventChannel.EventSink events) {
        eventSink = events;
    }

    @Override
    public void onCancel(Object arguments) {
        eventSink = null;
    }

    public void sink(List<Map<String, Object>> objects) {
        handler.post(() -> {
            if (eventSink != null && !objects.isEmpty()) {
                eventSink.success(objects);
            }
        });
    }

    public void close() {
        if (eventSink != null) {
            eventSink.endOfStream();
            eventSink = null;
        }
    }
}
