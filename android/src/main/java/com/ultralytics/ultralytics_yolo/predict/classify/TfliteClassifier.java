package com.ultralytics.ultralytics_yolo.predict.classify;

import static com.ultralytics.ultralytics_yolo.CameraPreview.CAMERA_PREVIEW_SIZE;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.os.Handler;
import android.os.Looper;

import androidx.camera.core.ImageProxy;

import com.ultralytics.ultralytics_yolo.ImageUtils;
import com.ultralytics.ultralytics_yolo.predict.PredictorException;
import com.ultralytics.ultralytics_yolo.models.LocalYoloModel;
import com.ultralytics.ultralytics_yolo.models.YoloModel;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.gpu.GpuDelegateFactory;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class TfliteClassifier extends Classifier {

    private static final long FPS_INTERVAL_MS = 1000; // Update FPS every 1000 milliseconds (1 second)
    private static final int NUM_BYTES_PER_CHANNEL = 4;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private long lastFpsTime = System.currentTimeMillis();
    private int frameCount = 0;
    private Interpreter interpreter;
    private Object[] inputArray;
    private int outputShape2;
    private Map<Integer, Object> outputMap;
    private ClassificationResultCallback classificationResultCallback;
    private FloatResultCallback inferenceTimeCallback;
    private FloatResultCallback fpsRateCallback;
    private final Matrix transformationMatrix;
    private final Bitmap pendingBitmapFrame;

    public TfliteClassifier(Context context) {
        super(context);

        pendingBitmapFrame = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        transformationMatrix = ImageUtils.getTransformationMatrix(CAMERA_PREVIEW_SIZE.getWidth(), CAMERA_PREVIEW_SIZE.getHeight(),
                INPUT_SIZE, INPUT_SIZE,
                90, false);
    }

    @Override
    public void setClassificationResultCallback(ClassificationResultCallback callback) {
        classificationResultCallback = callback;
    }

    @Override
    public void setNumItemsThreshold(int numItems) {

    }

    @Override
    public void loadModel(YoloModel yoloModel, boolean useGpu) throws Exception {
        if (yoloModel instanceof LocalYoloModel) {
            final LocalYoloModel localYoloModel = (LocalYoloModel) yoloModel;

            if (localYoloModel.modelPath == null || localYoloModel.modelPath.isEmpty() ||
                    localYoloModel.metadataPath == null || localYoloModel.metadataPath.isEmpty()) {
                throw new Exception();
            }

            final AssetManager assetManager = context.getAssets();
            loadLabels(assetManager, localYoloModel.metadataPath);
            try {
                MappedByteBuffer modelFile = loadModelFile(assetManager, localYoloModel.modelPath);
                initDelegate(modelFile, useGpu);
            } catch (Exception e) {
                throw new PredictorException("Error model");
            }
        }
    }

    @Override
    public List<ClassificationResult> predict(Bitmap bitmap) {
        try {
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
            setInput(resizedBitmap);
            return runInference();
        } catch (Exception e) {
            return new ArrayList<>();
        }
    }

    @Override
    public void setConfidenceThreshold(float confidence) {
    }

    @Override
    public void setInferenceTimeCallback(FloatResultCallback callback) {
        inferenceTimeCallback = callback;
    }

    @Override
    public void setFpsRateCallback(FloatResultCallback callback) {
        fpsRateCallback = callback;
    }

    private void initDelegate(MappedByteBuffer buffer, boolean useGpu) {
        Interpreter.Options interpreterOptions = new Interpreter.Options();
        try {
            // Check if GPU support is available
            CompatibilityList compatibilityList = new CompatibilityList();
            if (useGpu && compatibilityList.isDelegateSupportedOnThisDevice()) {
                GpuDelegateFactory.Options delegateOptions = compatibilityList.getBestOptionsForThisDevice();
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions.setQuantizedModelsAllowed(true));
                interpreterOptions.addDelegate(gpuDelegate);
            } else {
            interpreterOptions.setNumThreads(4);
            }
            // Create the interpreter
            this.interpreter = new Interpreter(buffer, interpreterOptions);
        } catch (Exception e) {
            interpreterOptions = new Interpreter.Options();
            interpreterOptions.setNumThreads(4);
            // Create the interpreter
            this.interpreter = new Interpreter(buffer, interpreterOptions);
        }

        int[] outputShape = interpreter.getOutputTensor(0).shape();
        outputShape2 = outputShape[1];
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // Local model from Flutter project
        if (modelPath.startsWith("flutter_assets")) {
            AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
        // Absolute path
        else {
            FileInputStream inputStream = new FileInputStream(modelPath);
            FileChannel fileChannel = inputStream.getChannel();
            long declaredLength = fileChannel.size();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, declaredLength);
        }
    }

    public void predict(ImageProxy imageProxy, boolean isMirrored) {
        if (interpreter == null || imageProxy == null) {
            return;
        }

        Bitmap bitmap = ImageUtils.toBitmap(imageProxy);
        Canvas canvas = new Canvas(pendingBitmapFrame);
        Matrix cropToFrameTransform = new Matrix();
        transformationMatrix.invert(cropToFrameTransform);
        canvas.drawBitmap(bitmap, transformationMatrix, null);

        handler.post(() -> {
            setInput(pendingBitmapFrame);

            long start = System.currentTimeMillis();
            List<ClassificationResult> result = runInference();
            long end = System.currentTimeMillis();

            // Increment frame count
            frameCount++;

            // Check if it's time to update FPS
            long elapsedMillis = end - lastFpsTime;
            if (elapsedMillis > FPS_INTERVAL_MS) {
                // Calculate frames per second
                float fps = (float) frameCount / elapsedMillis * 1000.f;

                // Reset counters for the next interval
                lastFpsTime = end;
                frameCount = 0;

                // Log or display the FPS
                fpsRateCallback.onResult(fps);
            }

            classificationResultCallback.onResult(result);
            inferenceTimeCallback.onResult(end - start);
        });
    }

    private void setInput(Bitmap resizedbitmap) {
        ByteBuffer imgData = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * NUM_BYTES_PER_CHANNEL);
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];

        resizedbitmap.getPixels(intValues, 0, resizedbitmap.getWidth(), 0, 0, resizedbitmap.getWidth(), resizedbitmap.getHeight());

        imgData.order(ByteOrder.nativeOrder());
        imgData.rewind();
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < INPUT_SIZE; ++j) {
                int pixelValue = intValues[i * INPUT_SIZE + j];
                float r = (((pixelValue >> 16) & 0xFF)) / 255.0f;
                float g = (((pixelValue >> 8) & 0xFF)) / 255.0f;
                float b = ((pixelValue & 0xFF)) / 255.0f;
                imgData.putFloat(r);
                imgData.putFloat(g);
                imgData.putFloat(b);
            }
        }
        this.inputArray = new Object[]{imgData};
        this.outputMap = new HashMap<>();
        ByteBuffer outData = ByteBuffer.allocateDirect(outputShape2 * NUM_BYTES_PER_CHANNEL);
        outData.order(ByteOrder.nativeOrder());
        outData.rewind();
        outputMap.put(0, outData);
    }

    private List<ClassificationResult> runInference() {
        List<ClassificationResult> classificationResults = new ArrayList<>();

        if (interpreter != null) {
            interpreter.runForMultipleInputsOutputs(inputArray, outputMap);

            ByteBuffer byteBuffer = (ByteBuffer) outputMap.get(0);
            if (byteBuffer != null) {
                byteBuffer.rewind();

                final List<HashMap<String, Object>> list = new ArrayList<>();
                for (int j = 0; j < outputShape2; ++j) {
                    final HashMap<String, Object> map = new HashMap<>();
                    float confidence = byteBuffer.getFloat();
                    String label = labels.get(j);

                    map.put("index", j);
                    map.put("label", label);
                    map.put("confidence", confidence);
                    list.add(map);
                }

                classificationResults = list.stream()
                        .sorted((map1, map2) -> {
                            float confidence1 = (float) map1.get("confidence");
                            float confidence2 = (float) map2.get("confidence");
                            return Float.compare(confidence2, confidence1);
                        })
                        .map(map -> new ClassificationResult((String) map.get("label"), (int) map.get("index"), (float) map.get("confidence")))
                        .collect(Collectors.toList());

            }
        }

        return classificationResults;
    }
}
