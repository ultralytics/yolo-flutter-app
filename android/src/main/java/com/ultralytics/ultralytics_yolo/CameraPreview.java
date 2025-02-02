package com.ultralytics.ultralytics_yolo;

import android.app.Activity;
import android.content.Context;
import android.util.Size;

import androidx.camera.core.AspectRatio;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraControl;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;
import com.ultralytics.ultralytics_yolo.predict.Predictor;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;


public class CameraPreview {
    public final static Size CAMERA_PREVIEW_SIZE = new Size(640, 480);
    private final Context context;
    private Predictor predictor;
    private ProcessCameraProvider cameraProvider;
    private CameraControl cameraControl;
    private Activity activity;
    private PreviewView mPreviewView;
    private boolean busy = false;
    private boolean deferredProcessing = false;

    public CameraPreview(Context context) {
        this.context = context;
    }

    public void openCamera(int facing, Activity activity, PreviewView mPreviewView, boolean deferredProcessing) {
        this.activity = activity;
        this.mPreviewView = mPreviewView;
        this.deferredProcessing = deferredProcessing;

        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(context);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindPreview(facing);
            } catch (ExecutionException | InterruptedException e) {
                // No errors need to be handled for this Future.
                // This should never be reached.
            }
        }, ContextCompat.getMainExecutor(context));
    }

    private void bindPreview(int facing) {
        if (!busy) {
            busy = true;

            final boolean isMirrored = (facing == CameraSelector.LENS_FACING_FRONT);

            Preview cameraPreview = new Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .build();

            CameraSelector cameraSelector = new CameraSelector.Builder()
                    .requireLensFacing(facing)
                    .build();

            ImageAnalysis imageAnalysis =
                    new ImageAnalysis.Builder()
                            .setBackpressureStrategy(deferredProcessing ? ImageAnalysis.STRATEGY_BLOCK_PRODUCER : ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                            .build();

            if (deferredProcessing) {
                final ExecutorService executorService = Executors.newSingleThreadExecutor();
                final AtomicBoolean isPredicting = new AtomicBoolean(false);

                imageAnalysis.setAnalyzer(Runnable::run, imageProxy -> {
                    if (isPredicting.get()) {
                        imageProxy.close();
                        return;
                    }

                    isPredicting.set(true);

                    executorService.submit(() -> {
                        try {
                            predictor.predict(imageProxy, isMirrored);
                        } catch (Exception e) {
                          e.printStackTrace();
                        } finally {
                            //clear stream for next image
                            imageProxy.close();

                            isPredicting.set(false);
                        }
                    });
                });
            } else {
                imageAnalysis.setAnalyzer(Runnable::run, imageProxy -> {
                    predictor.predict(imageProxy, isMirrored);
                    //clear stream for next image
                    imageProxy.close();
                });
            }

            // Unbind use cases before rebinding
            cameraProvider.unbindAll();

            // Bind use cases to camera
            Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner) activity, cameraSelector, cameraPreview, imageAnalysis);

            cameraControl = camera.getCameraControl();

            cameraPreview.setSurfaceProvider(mPreviewView.getSurfaceProvider());

            busy = false;
        }
    }

    public void setPredictorFrameProcessor(Predictor predictor) {
        this.predictor = predictor;
    }

    public void setCameraFacing(int facing) {
        if (cameraProvider != null) {
            cameraProvider.unbindAll();
            bindPreview(facing);
        }
    }

    public void setScaleFactor(double factor) {
        cameraControl.setZoomRatio((float)factor);
    }
}
