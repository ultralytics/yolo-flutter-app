// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.ultralytics_yolo.predict.segment;

import static com.ultralytics.ultralytics_yolo.CameraPreview.CAMERA_PREVIEW_SIZE;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.camera.core.ImageProxy;

import com.ultralytics.ultralytics_yolo.ImageUtils;
import com.ultralytics.ultralytics_yolo.models.LocalYoloModel;
import com.ultralytics.ultralytics_yolo.models.YoloModel;
import com.ultralytics.ultralytics_yolo.predict.PredictorException;

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
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.util.List;


public class TfliteSegmenter extends Segmenter {

    static {
        System.loadLibrary("ultralytics");
    }

    private static final long FPS_INTERVAL_MS = 1000; // Update FPS every 1000 milliseconds (1 second)
    private static final int NUM_BYTES_PER_CHANNEL = 4;
    private final Handler handler = new Handler(Looper.getMainLooper());
    private final Matrix transformationMatrix;
    private final Bitmap pendingBitmapFrame;
    private int numClasses;
    private int frameCount = 0;
    private double confidenceThreshold = 0.25f;
    private double iouThreshold = 0.45f;
    private int numItemsThreshold = 30;
    private Interpreter interpreter;
    private Object[] inputArray;
    private int outputShape0; // Batch size (always 1 for single image inference)
    private int outputShape1; // Number of detections
    private int outputShape2; 
     private int outputShape_m0; 
    private int outputShape_m1; 
    private int outputShape_m2;
    private int outputShape_m3; 
    private int maskChannels; // Number of channels in the mask output
    private float[][][] output; // [1][num_detections][4 + 1 + num_classes + mask_channels]
    private long lastFpsTime = System.currentTimeMillis();
    private Map<Integer, Object> outputMap;
    private SegmentationResultCallback segmentationResultCallback;
    private FloatResultCallback inferenceTimeCallback;
    private FloatResultCallback fpsRateCallback;
    private static final String TAG = "TfliteSegmenter";

    // Declare rawDetections and maskProtos as class members
    private float[][][] rawDetections;
    private float[][][][] maskProtos = new float[1][160][160][32];

    public TfliteSegmenter(Context context) {
        super(context);
        pendingBitmapFrame = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        transformationMatrix = new Matrix();
    }

    @Override
    public void loadModel(YoloModel yoloModel, boolean useGpu) throws Exception {
        
        if (yoloModel instanceof LocalYoloModel) {
            final LocalYoloModel localYoloModel = (LocalYoloModel) yoloModel;

            if (localYoloModel.modelPath == null || localYoloModel.modelPath.isEmpty() ||
                    localYoloModel.metadataPath == null || localYoloModel.metadataPath.isEmpty()) {
                throw new Exception("Model or metadata path cannot be null or empty.");
            }

            final AssetManager assetManager = context.getAssets();
            loadLabels(assetManager, localYoloModel.metadataPath);
            numClasses = labels.size();
            try {
                MappedByteBuffer modelFile = loadModelFile(assetManager, localYoloModel.modelPath);
                initDelegate(modelFile, useGpu);
            } catch (IOException e) {
                System.out.println("ERROR:"+ e.getMessage());
                throw new PredictorException("Error loading model file: " + e.getMessage());
            }
        } else {
            throw new IllegalArgumentException("Only LocalYoloModel is supported for TfliteSegmenter.");
        }
    }

    @Override
    public Object[] predict(Bitmap bitmap) {
        
        try {
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
            setInput(resizedBitmap);
            return runInference(); // Now correctly returning float[][]
        } catch (Exception e) {
            return new Object[0];
        }
    }

    @Override
    public void setConfidenceThreshold(float confidence) {
        this.confidenceThreshold = confidence;
    }

    @Override
    public void setIouThreshold(float iou) {
        this.iouThreshold = iou;
    }

    @Override
    public void setNumItemsThreshold(int numItems) {
        this.numItemsThreshold = numItems;
    }

    @Override
    public void setSegmentationResultCallback(SegmentationResultCallback callback) {
        this.segmentationResultCallback = callback;
    }

    @Override
    public void setInferenceTimeCallback(FloatResultCallback callback) {
        this.inferenceTimeCallback = callback;
    }

    @Override
    public void setFpsRateCallback(FloatResultCallback callback) {
        this.fpsRateCallback = callback;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
     
        if (modelPath.startsWith("flutter_assets")) {
            AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        } else {
            FileInputStream inputStream = new FileInputStream(modelPath);
            FileChannel fileChannel = inputStream.getChannel();
            long declaredLength = fileChannel.size();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, declaredLength);
        }
    }

    private void initDelegate(MappedByteBuffer buffer, boolean useGpu) {
         
        Interpreter.Options interpreterOptions = new Interpreter.Options();
        try {
            CompatibilityList compatibilityList = new CompatibilityList();
            if (useGpu && compatibilityList.isDelegateSupportedOnThisDevice()) {
                GpuDelegateFactory.Options delegateOptions = compatibilityList.getBestOptionsForThisDevice();
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions.setQuantizedModelsAllowed(true));
                interpreterOptions.addDelegate(gpuDelegate);
            } else {
                interpreterOptions.setNumThreads(4);
            }
            this.interpreter = new Interpreter(buffer, interpreterOptions);
        } catch (Exception e) {
            Log.e(TAG,"deligate error==========="+e);
            interpreterOptions = new Interpreter.Options();
            interpreterOptions.setNumThreads(4);
            this.interpreter = new Interpreter(buffer, interpreterOptions);
        }
 
        int[] outputShape = interpreter.getOutputTensor(0).shape();
        int[] outputShape_m = interpreter.getOutputTensor(1).shape();
        outputShape0 = outputShape[0]; // Should be 1
        outputShape1 = outputShape[1]; // Number of detections
        outputShape2 = outputShape[2]; // 4 (bbox) + 1 (confidence) + num_classes + mask_channels
outputShape_m0 =outputShape_m[0];
outputShape_m1=outputShape_m[1];
outputShape_m2=outputShape_m[2];
outputShape_m3=outputShape_m[3];
        // Assuming the mask channels are the last part of the output
        // We need to infer the number of mask channels based on the output shape
        maskChannels = outputShape1 - 4  - numClasses;
        if (maskChannels <= 0) {
            throw new IllegalStateException("Invalid output shape for segmentation model. " +
                    "Expected at least 4 (bbox) + 1 (confidence) + num_classes + >0 (mask channels). " +
                    "Got output shape: [" + outputShape0 + ", " + outputShape1 + ", " + outputShape2 + "]");
        }

        output = new float[outputShape0][outputShape1][outputShape2];
        rawDetections = new float[outputShape0][outputShape1][outputShape2];
    }


    public void predict(ImageProxy imageProxy, boolean isMirrored) {
         
        if (interpreter == null || imageProxy == null) {
            return;
        }

        Bitmap bitmap = ImageUtils.toBitmap(imageProxy);
        Canvas canvas = new Canvas(pendingBitmapFrame);

        // Calculate transformation based on orientation and mirroring
        transformationMatrix.reset();

        // Handle rotation based on image rotation
        float rotation = imageProxy.getImageInfo().getRotationDegrees();
        float centerX = INPUT_SIZE / 2f;
        float centerY = INPUT_SIZE / 2f;

        transformationMatrix.postRotate(rotation, centerX, centerY);

        // Handle mirroring for front camera (applied to the bitmap)
        if (isMirrored) {
            transformationMatrix.postScale(-1, 1, centerX, centerY);
        }

        // Scale the image to fit INPUT_SIZE
        float scaleX = (float) INPUT_SIZE / bitmap.getWidth();
        float scaleY = (float) INPUT_SIZE / bitmap.getHeight();
        float scale = Math.max(scaleX, scaleY);
        transformationMatrix.postScale(scale, scale, centerX, centerY);

        // Center the image
        float dx = centerX - (bitmap.getWidth() * scale) / 2;
        float dy = centerY - (bitmap.getHeight() * scale) / 2;
        transformationMatrix.postTranslate(dx, dy);

        canvas.drawBitmap(bitmap, transformationMatrix, null);

        handler.post(() -> {
            setInput(pendingBitmapFrame);

            long start = System.currentTimeMillis();
            Object[] rawResult = runInference(); // result is now Object[]

            List<HashMap<String, Object>> results = new ArrayList<>();
            if (rawResult != null) {
                for (Object item : rawResult) {
                    if (item instanceof HashMap) {
                        results.add((HashMap<String, Object>) item);
                    } else {
                        Log.e(TAG, "Unexpected item type in inference result: " + item.getClass().getName());
                    }
                }
            }

            // If front camera, flip the x coordinates of the bounding boxes and polygons
            if (isMirrored) {
                for (HashMap<String, Object> detectionMap : results) {
                    // Flip bounding box x-coordinates
                    Float x = (Float) detectionMap.get("x");
                    Float width = (Float) detectionMap.get("width");
                    if (x != null && width != null) {
                        detectionMap.put("x", 1.0f - x - width);
                    }

                    // Flip polygon x-coordinates
                    List<List<HashMap<String, Integer>>> polygons = (List<List<HashMap<String, Integer>>>) detectionMap.get("polygons");
                    if (polygons != null) {
                        List<List<HashMap<String, Integer>>> flippedPolygons = new ArrayList<>();
                        for (List<HashMap<String, Integer>> polygon : polygons) {
                            List<HashMap<String, Integer>> flippedPolygon = new ArrayList<>();
                            for (HashMap<String, Integer> point : polygon) {
                                Integer pointX = point.get("x");
                                Integer pointY = point.get("y");
                                if (pointX != null) {
                                    HashMap<String, Integer> flippedPoint = new HashMap<>();
                                    flippedPoint.put("x", (int) (1.0f - (float) pointX));
                                    flippedPoint.put("y", pointY);
                                    flippedPolygon.add(flippedPoint);
                                } else {
                                    flippedPolygon.add(point); // Keep as is if x is null (shouldn't happen)
                                }
                            }
                            flippedPolygons.add(flippedPolygon);
                        }
                        detectionMap.put("polygons", flippedPolygons);
                    }
                }
            }

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
                if (fpsRateCallback != null) {
                    fpsRateCallback.onResult(fps);
                }
            }

            if (segmentationResultCallback != null) {
                
                segmentationResultCallback.onResult(results.toArray()); // Pass the Object[]
            }
            if (inferenceTimeCallback != null) {
                inferenceTimeCallback.onResult(end - start);
            }
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
        ByteBuffer outData = ByteBuffer.allocateDirect(outputShape0 * outputShape1 * outputShape2 * NUM_BYTES_PER_CHANNEL);
        outData.order(ByteOrder.nativeOrder());
        outData.rewind();
        outputMap.put(0, outData);
        outputMap.put(1, ByteBuffer.allocateDirect(outputShape0 * outputShape_m1 * outputShape_m2 * outputShape_m3 * NUM_BYTES_PER_CHANNEL).order(ByteOrder.nativeOrder()));
    }

    private Object[] runInference() {
      
        if (interpreter != null) {
            interpreter.runForMultipleInputsOutputs(inputArray, outputMap);

            ByteBuffer byteBuffer = (ByteBuffer) outputMap.get(0);
            ByteBuffer byteBuffer_mask = (ByteBuffer) outputMap.get(1);
            

            if (byteBuffer != null && byteBuffer_mask != null) {
                byteBuffer.rewind();
               
                for (int i = 0; i < 1; ++i) {
                    for (int j = 0; j < outputShape1; ++j) {
                        for (int k = 0; k < outputShape2; ++k) {
                            rawDetections[i][j][k] = byteBuffer.getFloat();
                        }
                    }
                }

                byteBuffer_mask.rewind();
                for (int b = 0; b < 1; ++b) {
                    for (int h = 0; h < outputShape_m1; ++h) {
                        for (int w = 0; w < outputShape_m2; ++w) {
                            for (int c = 0; c < outputShape_m3; ++c) {
                                maskProtos[b][h][w][c] = byteBuffer_mask.getFloat();
                            }
                        }
                    }
                }

             
                
            } else {
                Log.e(TAG, "Error: byteBuffer or byteBuffer_mask is null.");
                return new float[0][];
            }

            // Add null checks before calling native method
            if (rawDetections == null || maskProtos == null) {
                Log.e(TAG, "Error: rawDetections or maskProtos is null.");
                return new float[0][];
            }
            return postprocess(rawDetections, maskProtos, outputShape2, outputShape1,
                    (float) confidenceThreshold, (float) iouThreshold,
                    numItemsThreshold, numClasses, maskChannels,outputShape_m0, outputShape_m1,outputShape_m2,outputShape_m3);
        }
        return new Object[0];
    }

 public native Object[] postprocess(
    float[][][] raw_detections_obj,
    float[][][][] mask_protos_obj,
    int w, int h,
    float confidence_threshold,
    float iou_threshold,
    int num_items_threshold,
    int num_classes,
    int mask_channels,
    int mask_shape0,
    int mask_shape1,
    int mask_shape2,
    int mask_shape3
);
}