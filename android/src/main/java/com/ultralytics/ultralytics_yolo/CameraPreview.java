package com.ultralytics.ultralytics_yolo;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.util.Size;

import androidx.camera.core.AspectRatio;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraControl;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;
import com.ultralytics.ultralytics_yolo.predict.Predictor;

import java.io.ByteArrayOutputStream;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;


public class CameraPreview {
    public final static Size CAMERA_PREVIEW_SIZE = new Size(640, 480);
    private final Context context;
    private Predictor predictor;
    private ProcessCameraProvider cameraProvider;
    private CameraControl cameraControl;
    private Activity activity;
    private PreviewView mPreviewView;
    private boolean busy = false;
    private boolean shouldCaptureFrame = false;
    BlockingQueue<byte[]> capturedFrameQueue = new LinkedBlockingQueue<>();

    public CameraPreview(Context context) {
        this.context = context;
    }

    public void openCamera(int facing, Activity activity, PreviewView mPreviewView) {
        this.activity = activity;
        this.mPreviewView = mPreviewView;

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

            Preview cameraPreview = new Preview.Builder()
                    .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                    .build();

            CameraSelector cameraSelector = new CameraSelector.Builder()
                    .requireLensFacing(facing)
                    .build();

            ImageAnalysis imageAnalysis =
                    new ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                            .build();
            imageAnalysis.setAnalyzer(Runnable::run, imageProxy -> {
                predictor.predict(imageProxy, facing == CameraSelector.LENS_FACING_FRONT);

                if (shouldCaptureFrame) {
                    shouldCaptureFrame = false;

                    try {
                        final byte[] data = toCapturedFrameData(imageProxy);
                        capturedFrameQueue.put(data);
                    } catch (InterruptedException ex) {
                        ex.printStackTrace();
                    }
                }

                //clear stream for next image
                imageProxy.close();
            });

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
        bindPreview(facing);
    }

    public void setScaleFactor(double factor) {
        cameraControl.setZoomRatio((float)factor);
    }

    public CompletableFuture<byte[]> requestCaptureVideo(int timeoutSec) {
        this.shouldCaptureFrame = true;

        CompletableFuture<byte[]> future = new CompletableFuture<>();

        CompletableFuture.runAsync(() -> {
            try {
                final byte[] captured = capturedFrameQueue.poll(timeoutSec, TimeUnit.SECONDS);
                if (captured != null) {
                    future.complete(captured);
                } else {
                    future.completeExceptionally(new Error("Buffer is null"));
                }
            } catch (InterruptedException e) {
                future.completeExceptionally(e);
            }
        });

        return future;
    }

    private byte[] toCapturedFrameData(ImageProxy imageProxy) {
        Bitmap bitmap = ImageUtils.toBitmap(imageProxy);

        final Bitmap outputBitmap = Bitmap.createBitmap(bitmap.getHeight(), bitmap.getWidth(), Bitmap.Config.ARGB_8888);
        final Matrix transformationMatrix = ImageUtils.getTransformationMatrix(bitmap.getWidth(), bitmap.getHeight(),
                bitmap.getHeight(), bitmap.getWidth(),
                90, false);

        Canvas canvas = new Canvas(outputBitmap);
        canvas.drawBitmap(bitmap, transformationMatrix, null);

        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        outputBitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);

        return outputStream.toByteArray();
    }
}
