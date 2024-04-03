package com.ultralytics.ultralytics_yolo;

import android.app.Activity;
import android.content.Context;

import androidx.annotation.NonNull;

import io.flutter.embedding.engine.plugins.FlutterPlugin;
import io.flutter.embedding.engine.plugins.activity.ActivityAware;
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding;
import io.flutter.plugin.common.BinaryMessenger;
import io.flutter.plugin.common.MethodChannel;

/**
 * UltralyticsYoloPlugin
 */
public class UltralyticsYoloPlugin implements FlutterPlugin, ActivityAware {
    private FlutterPluginBinding flutterPluginBinding;
    private CameraPreview cameraPreview;

    @Override
    public void onAttachedToEngine(@NonNull FlutterPluginBinding binding) {
        flutterPluginBinding = binding;

        BinaryMessenger binaryMessenger = flutterPluginBinding.getBinaryMessenger();
        Context context = flutterPluginBinding.getApplicationContext();

        cameraPreview = new CameraPreview(context);

        MethodCallHandler methodCallHandler = new MethodCallHandler(
                binaryMessenger,
                context, cameraPreview);
        new MethodChannel(binaryMessenger, "ultralytics_yolo")
                .setMethodCallHandler(methodCallHandler);
    }

    @Override
    public void onDetachedFromEngine(@NonNull FlutterPluginBinding binding) {
        flutterPluginBinding = null;
    }

    @Override
    public void onAttachedToActivity(@NonNull ActivityPluginBinding binding) {
        Activity activity = binding.getActivity();
        NativeViewFactory nativeViewFactory = new NativeViewFactory(activity, cameraPreview);
        flutterPluginBinding
                .getPlatformViewRegistry()
                .registerViewFactory("ultralytics_yolo_camera_preview", nativeViewFactory);
    }

    @Override
    public void onDetachedFromActivityForConfigChanges() {

    }

    @Override
    public void onReattachedToActivityForConfigChanges(@NonNull ActivityPluginBinding binding) {
        onAttachedToActivity(binding);
    }

    @Override
    public void onDetachedFromActivity() {

    }
}