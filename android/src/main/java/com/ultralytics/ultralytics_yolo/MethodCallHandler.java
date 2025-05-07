// // Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// package com.ultralytics.ultralytics_yolo;

// import static com.ultralytics.ultralytics_yolo.CameraPreview.CAMERA_PREVIEW_SIZE;

// import android.content.Context;
// import android.graphics.Bitmap;
// import android.graphics.BitmapFactory;
// import android.util.DisplayMetrics;

// import androidx.annotation.NonNull;

// import com.ultralytics.ultralytics_yolo.models.LocalYoloModel;
// import com.ultralytics.ultralytics_yolo.models.RemoteYoloModel;
// import com.ultralytics.ultralytics_yolo.models.YoloModel;
// import com.ultralytics.ultralytics_yolo.predict.Predictor;
// import com.ultralytics.ultralytics_yolo.predict.classify.ClassificationResult;
// import com.ultralytics.ultralytics_yolo.predict.classify.Classifier;
// import com.ultralytics.ultralytics_yolo.predict.classify.TfliteClassifier;
// import com.ultralytics.ultralytics_yolo.predict.detect.Detector;
// import com.ultralytics.ultralytics_yolo.predict.detect.TfliteDetector;

// import java.util.ArrayList;
// import java.util.HashMap;
// import java.util.List;
// import java.util.Map;
// import java.util.Objects;

// import io.flutter.plugin.common.BinaryMessenger;
// import io.flutter.plugin.common.EventChannel;
// import io.flutter.plugin.common.MethodCall;
// import io.flutter.plugin.common.MethodChannel;

// public class MethodCallHandler implements MethodChannel.MethodCallHandler {
//     private final Context context;
//     private final CameraPreview cameraPreview;
//     private Predictor predictor;
//     private final ResultStreamHandler resultStreamHandler;
//     private final InferenceTimeStreamHandler inferenceTimeStreamHandler;
//     private final FpsRateStreamHandler fpsRateStreamHandler;
//     private final float widthDp;
//     private final float density;
//     private final float heightDp;

//     public MethodCallHandler(BinaryMessenger binaryMessenger, Context context, CameraPreview cameraPreview) {
//         this.context = context;

//         this.cameraPreview = cameraPreview;

//         EventChannel predictionResultEventChannel = new EventChannel(binaryMessenger, "ultralytics_yolo_prediction_results");
//         resultStreamHandler = new ResultStreamHandler();
//         predictionResultEventChannel.setStreamHandler(resultStreamHandler);

//         EventChannel inferenceTimeEventChannel = new EventChannel(binaryMessenger, "ultralytics_yolo_inference_time");
//         inferenceTimeStreamHandler = new InferenceTimeStreamHandler();
//         inferenceTimeEventChannel.setStreamHandler(inferenceTimeStreamHandler);

//         EventChannel fpsRateEventChannel = new EventChannel(binaryMessenger, "ultralytics_yolo_fps_rate");
//         fpsRateStreamHandler = new FpsRateStreamHandler();
//         fpsRateEventChannel.setStreamHandler(fpsRateStreamHandler);


//         DisplayMetrics displayMetrics = context.getResources().getDisplayMetrics();
//         int widthPixels = displayMetrics.widthPixels;
//         int heightPixels = displayMetrics.heightPixels;
//         density = displayMetrics.density;
//         widthDp = widthPixels / density;
//         // Add 40dp to resolve the discrepancy between Flutter screen and AndroidView
//         // caused by the presence of the navigation bar
//         heightDp = heightPixels / density + 40;
//     }

//     @Override
//     public void onMethodCall(@NonNull MethodCall call, @NonNull MethodChannel.Result result) {
//         String method = call.method;
//         switch (method) {
//             case "loadModel":
//                 loadModel(call, result);
//                 break;
//             case "setConfidenceThreshold":
//                 setConfidenceThreshold(call, result);
//                 break;
//             case "setIouThreshold":
//                 setIouThreshold(call, result);
//                 break;
//             case "setNumItemsThreshold":
//                 setNumItemsThreshold(call, result);
//                 break;
//             case "detectImage":
//                 detectImage(call, result);
//                 break;
//             case "classifyImage":
//                 classifyImage(call, result);
//                 break;
//             case "setLensDirection":
//                 setLensDirection(call, result);
//                 break;
//             case "closeCamera":
//                 closeCamera(call, result);
//                 break;
//             case "startCamera":
//                 startCamera(call, result);
//                 break;
//             case "pauseLivePrediction":
//                 pauseLivePrediction(call, result);
//                 break;
//             case "resumeLivePrediction":
//                 resumeLivePrediction(call, result);
//                 break;
//             case "setZoomRatio":
//                 setScaleFactor(call, result);
//                 break;
//             default:
//                 result.notImplemented();
//                 break;
//         }
//     }

//     private void loadModel(MethodCall call, MethodChannel.Result result) {
//         Map<String, Object> model = call.argument("model");
//         if (model == null) {
//             result.error("PredictorError", "Invalid model", null);
//             return;
//         }

//         YoloModel yoloModel = null;
//         String type = (String) model.get("type");
//         String task = (String) model.get("task");
//         String format = (String) model.get("format");
//         if (Objects.equals(task, "detect")) {
//             if (Objects.equals(format, "tflite")) {
//                 predictor = new TfliteDetector(context);
//             }
//         } else if (Objects.equals(task, "classify")) {
//             if (Objects.equals(format, "tflite")) {
//                 predictor = new TfliteClassifier(context);
//             }
//         } else {
//             return;
//         }

//         switch (Objects.requireNonNull(type)) {
//             case "local":
//                 String modelPath = (String) model.get("modelPath");
//                 String metadataPath = (String) model.get("metadataPath");

//                 yoloModel = new LocalYoloModel(task, format, modelPath, metadataPath);
//                 break;
//             case "remote":
//                 String modelUrl = (String) model.get("modelUrl");

//                 yoloModel = new RemoteYoloModel(modelUrl, task);
//                 break;
//         }

//         try {
//             Object useGpuObject = call.argument("useGpu");
//             boolean useGpu = false;
//             if (useGpuObject != null) {
//                 useGpu = (boolean) useGpuObject;
//             }

//             predictor.loadModel(yoloModel, true);

//             setPredictorFrameProcessor();
//             setPredictorCallbacks();

//             result.success("Success");
//         } catch (Exception e) {
//             result.error("PredictorError", "Invalid model", null);
//         }
//     }

//     private void setPredictorFrameProcessor() {
//         cameraPreview.setPredictorFrameProcessor(predictor);
//     }

//     private void setPredictorCallbacks() {
//         if (predictor instanceof Detector) {
//             // Multiply by 3/4 instead of 4/3 because the camera preview frame is rotated -90°
//             // float newWidth = heightDp * 3 / 4;
//             float newWidth = heightDp * CAMERA_PREVIEW_SIZE.getHeight() / CAMERA_PREVIEW_SIZE.getWidth();
//             final float offsetX = (widthDp - newWidth) / 2;

//             ((Detector) predictor).setObjectDetectionResultCallback(result -> {
//                 List<Map<String, Object>> objects = new ArrayList<>();

//                 for (float[] obj : result) {
//                     Map<String, Object> objectMap = new HashMap<>();

//                     float x = obj[0] * newWidth + offsetX;
//                     float y = obj[1] * heightDp;
//                     float width = obj[2] * newWidth;
//                     float height = obj[3] * heightDp;
//                     float confidence = obj[4];
//                     int index = (int) obj[5];
//                     String label = index < predictor.labels.size() ? predictor.labels.get(index) : "";

//                     objectMap.put("x", x);
//                     objectMap.put("y", y);
//                     objectMap.put("width", width);
//                     objectMap.put("height", height);
//                     objectMap.put("confidence", confidence);
//                     objectMap.put("index", index);
//                     objectMap.put("label", label);

//                     objects.add(objectMap);
//                 }

//                 resultStreamHandler.sink(objects);
//             });
//         } else if (predictor instanceof Classifier) {
//             ((Classifier) predictor).setClassificationResultCallback(result -> {
//                 List<Map<String, Object>> objects = new ArrayList<>();

//                 for (ClassificationResult classificationResult : result) {
//                     Map<String, Object> objectMap = new HashMap<>();

//                     objectMap.put("confidence", classificationResult.confidence);
//                     objectMap.put("index", classificationResult.index);
//                     objectMap.put("label", classificationResult.label);
//                     objects.add(objectMap);
//                 }

//                 resultStreamHandler.sink(objects);
//             });
//         }

//         predictor.setFpsRateCallback(fpsRateStreamHandler::sink);
//         predictor.setInferenceTimeCallback(inferenceTimeStreamHandler::sink);
//     }

//     private void setConfidenceThreshold(MethodCall call, MethodChannel.Result result) {
//         Object confidenceObject = call.argument("confidence");
//         if (confidenceObject != null) {
//             final double confidence = (double) confidenceObject;
//             predictor.setConfidenceThreshold((float) confidence);
//         }
//     }

//     private void setIouThreshold(MethodCall call, MethodChannel.Result result) {
//         Object iouObject = call.argument("iou");
//         if (iouObject != null) {
//             final double iou = (double) iouObject;
//             ((Detector) predictor).setIouThreshold((float) iou);
//         }
//     }

//     private void setNumItemsThreshold(MethodCall call, MethodChannel.Result result) {
//         Object numItemsObject = call.argument("numItems");
//         if (numItemsObject != null) {
//             final int numItems = (int) numItemsObject;
//             ((Detector) predictor).setNumItemsThreshold(numItems);
//         }
//     }

//     private void setLensDirection(MethodCall call, MethodChannel.Result result) {
//         Object directionObject = call.argument("direction");
//         if (directionObject != null) {
//             final int direction = (int) directionObject;
//             cameraPreview.setCameraFacing(direction);
//         }
//     }

//     private void closeCamera(MethodCall call, MethodChannel.Result result) {
// //        ncnnCameraPreview.closeCamera();
//     }

//     private void startCamera(MethodCall call, MethodChannel.Result result) {
//         // TODO: Resume detector
//         // startCamera(0);
//     }

//     private void pauseLivePrediction(MethodCall call, MethodChannel.Result result) {
// //        ncnnCameraPreview.pauseLivePrediction();
//     }

//     private void resumeLivePrediction(MethodCall call, MethodChannel.Result result) {
// //        ncnnCameraPreview.resumeLivePrediction();
//     }

//     private void detectImage(MethodCall call, MethodChannel.Result result) {
//         if (predictor != null) {
//             Object imagePathObject = call.argument("imagePath");
//             if (imagePathObject != null) {
//                 final String imagePath = (String) imagePathObject;
//                 Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
//                 final float[][] res = (float[][]) predictor.predict(bitmap);

//                 float scaleFactor = widthDp / bitmap.getWidth();
//                 float newHeight = bitmap.getHeight() * scaleFactor;
//                 List<Map<String, Object>> objects = new ArrayList<>();
//                 for (float[] obj : res) {
//                     Map<String, Object> objectMap = new HashMap<>();

//                     float x = obj[0] * widthDp;
//                     float y = obj[1] * newHeight;
//                     float width = obj[2] * widthDp;
//                     float height = obj[3] * newHeight;
//                     float confidence = obj[4];
//                     int index = (int) obj[5];
//                     String label = index < predictor.labels.size() ? predictor.labels.get(index) : "";

//                     objectMap.put("x", x);
//                     objectMap.put("y", y);
//                     objectMap.put("width", width);
//                     objectMap.put("height", height);
//                     objectMap.put("confidence", confidence);
//                     objectMap.put("index", index);
//                     objectMap.put("label", label);

//                     objects.add(objectMap);
//                 }

//                 result.success(objects);
//             }
//         }
//     }

// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.ultralytics_yolo;

import static com.ultralytics.ultralytics_yolo.CameraPreview.CAMERA_PREVIEW_SIZE;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.util.DisplayMetrics;
import android.util.Log;

import androidx.annotation.NonNull;

import com.ultralytics.ultralytics_yolo.models.LocalYoloModel;
import com.ultralytics.ultralytics_yolo.models.RemoteYoloModel;
import com.ultralytics.ultralytics_yolo.models.YoloModel;
import com.ultralytics.ultralytics_yolo.predict.Predictor;
import com.ultralytics.ultralytics_yolo.predict.classify.ClassificationResult;
import com.ultralytics.ultralytics_yolo.predict.classify.Classifier;
import com.ultralytics.ultralytics_yolo.predict.classify.TfliteClassifier;
import com.ultralytics.ultralytics_yolo.predict.detect.Detector;
import com.ultralytics.ultralytics_yolo.predict.detect.DetectedObject;
import com.ultralytics.ultralytics_yolo.predict.detect.TfliteDetector;
import com.ultralytics.ultralytics_yolo.predict.segment.DetectedSegment;
import com.ultralytics.ultralytics_yolo.predict.segment.Segmenter;
import com.ultralytics.ultralytics_yolo.predict.segment.TfliteSegmenter;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import io.flutter.plugin.common.BinaryMessenger;
import io.flutter.plugin.common.EventChannel;
import io.flutter.plugin.common.MethodCall;
import io.flutter.plugin.common.MethodChannel;

public class MethodCallHandler implements MethodChannel.MethodCallHandler {
    private final Context context;
    private final CameraPreview cameraPreview;
    private Predictor predictor;
    private final ResultStreamHandler resultStreamHandler;
    private final InferenceTimeStreamHandler inferenceTimeStreamHandler;
    private final FpsRateStreamHandler fpsRateStreamHandler;
    private final float widthDp;
    private final float density;
    private final float heightDp;
    private static final String TAG = "TfliteSegmenter";

    public MethodCallHandler(BinaryMessenger binaryMessenger, Context context, CameraPreview cameraPreview) {
        this.context = context;

        this.cameraPreview = cameraPreview;

        EventChannel predictionResultEventChannel = new EventChannel(binaryMessenger, "ultralytics_yolo_prediction_results");
        resultStreamHandler = new ResultStreamHandler();
        predictionResultEventChannel.setStreamHandler(resultStreamHandler);

        EventChannel inferenceTimeEventChannel = new EventChannel(binaryMessenger, "ultralytics_yolo_inference_time");
        inferenceTimeStreamHandler = new InferenceTimeStreamHandler();
        inferenceTimeEventChannel.setStreamHandler(inferenceTimeStreamHandler);

        EventChannel fpsRateEventChannel = new EventChannel(binaryMessenger, "ultralytics_yolo_fps_rate");
        fpsRateStreamHandler = new FpsRateStreamHandler();
        fpsRateEventChannel.setStreamHandler(fpsRateStreamHandler);


        DisplayMetrics displayMetrics = context.getResources().getDisplayMetrics();
        int widthPixels = displayMetrics.widthPixels;
        int heightPixels = displayMetrics.heightPixels;
        density = displayMetrics.density;
        widthDp = widthPixels / density;
        // Add 40dp to resolve the discrepancy between Flutter screen and AndroidView
        // caused by the presence of the navigation bar
        heightDp = heightPixels / density + 40;
    }

    @Override
    public void onMethodCall(@NonNull MethodCall call, @NonNull MethodChannel.Result result) {
        String method = call.method;
        switch (method) {
            case "loadModel":
                loadModel(call, result);
                break;
            case "setConfidenceThreshold":
                setConfidenceThreshold(call, result);
                break;
            case "setIouThreshold":
                setIouThreshold(call, result);
                break;
            case "setNumItemsThreshold":
                setNumItemsThreshold(call, result);
                break;
            case "detectImage":
                detectImage(call, result);
                break;
            case "classifyImage":
                classifyImage(call, result);
                break;
            case "segmentImage":
                segmentImage(call, result);
                break;
            case "setLensDirection":
                setLensDirection(call, result);
                break;
            case "closeCamera":
                closeCamera(call, result);
                break;
            case "startCamera":
                startCamera(call, result);
                break;
            case "pauseLivePrediction":
                pauseLivePrediction(call, result);
                break;
            case "resumeLivePrediction":
                resumeLivePrediction(call, result);
                break;
            case "setZoomRatio":
                setScaleFactor(call, result);
                break;
            default:
                result.notImplemented();
                break;
        }
    }

    private void loadModel(MethodCall call, MethodChannel.Result result) {
        Map<String, Object> model = call.argument("model");
        if (model == null) {
            result.error("PredictorError", "Invalid model", null);
            return;
        }

        YoloModel yoloModel = null;
        String type = (String) model.get("type");
        String task = (String) model.get("task");
        String format = (String) model.get("format");
        if (Objects.equals(task, "detect")) {
            if (Objects.equals(format, "tflite")) {
                predictor = new TfliteDetector(context);
            }
        } else if (Objects.equals(task, "classify")) {
            if (Objects.equals(format, "tflite")) {
                predictor = new TfliteClassifier(context);
            }
        } else if (Objects.equals(task, "segment")) {
            if (Objects.equals(format, "tflite")) {
                predictor = new TfliteSegmenter(context);
            }
        } else {
            return;
        }

        switch (Objects.requireNonNull(type)) {
            case "local":
                String modelPath = (String) model.get("modelPath");
                String metadataPath = (String) model.get("metadataPath");

                yoloModel = new LocalYoloModel(task, format, modelPath, metadataPath);
                break;
            case "remote":
                String modelUrl = (String) model.get("modelUrl");

                yoloModel = new RemoteYoloModel(modelUrl, task);
                break;
        }

        try {
            Object useGpuObject = call.argument("useGpu");
            boolean useGpu = false;
            if (useGpuObject != null) {
                useGpu = (boolean) useGpuObject;
            }

            predictor.loadModel(yoloModel, useGpu);

            setPredictorFrameProcessor();
            setPredictorCallbacks();

            result.success("Success");
        } catch (Exception e) {
            result.error("PredictorError", "Error loading model: " + e.getMessage(), null);
        }
    }

    private void setPredictorFrameProcessor() {
        cameraPreview.setPredictorFrameProcessor(predictor);
    }

    private void setPredictorCallbacks() {
        if (predictor instanceof Detector) {
            float newWidth = heightDp * CAMERA_PREVIEW_SIZE.getHeight() / CAMERA_PREVIEW_SIZE.getWidth();
            final float offsetX = (widthDp - newWidth) / 2;

            ((Detector) predictor).setObjectDetectionResultCallback(result -> {
                List<Map<String, Object>> objects = new ArrayList<>();

                for (float[] obj : result) {
                    Map<String, Object> objectMap = new HashMap<>();

                    float x = obj[0] * newWidth + offsetX;
                    float y = obj[1] * heightDp;
                    float width = obj[2] * newWidth;
                    float height = obj[3] * heightDp;
                    float confidence = obj[4];
                    int index = (int) obj[5];
                    String label = index < predictor.labels.size() ? predictor.labels.get(index) : "";

                    objectMap.put("x", x);
                    objectMap.put("y", y);
                    objectMap.put("width", width);
                    objectMap.put("height", height);
                    objectMap.put("confidence", confidence);
                    objectMap.put("index", index);
                    objectMap.put("label", label);

                    objects.add(objectMap);
                }

                resultStreamHandler.sink(objects);
            });
        } else if (predictor instanceof Classifier) {
            ((Classifier) predictor).setClassificationResultCallback(result -> {
                List<Map<String, Object>> objects = new ArrayList<>();

                for (ClassificationResult classificationResult : result) {
                    Map<String, Object> objectMap = new HashMap<>();

                    objectMap.put("confidence", classificationResult.confidence);
                    objectMap.put("index", classificationResult.index);
                    objectMap.put("label", classificationResult.label);
                    objects.add(objectMap);
                }

                resultStreamHandler.sink(objects);
            });
        } else if (predictor instanceof Segmenter) {
           
    float newWidth = heightDp * CAMERA_PREVIEW_SIZE.getHeight() / CAMERA_PREVIEW_SIZE.getWidth();
    final float offsetX = (widthDp - newWidth) / 2;

 ((Segmenter) predictor).setSegmentationResultCallback(results -> {
    List<Map<String, Object>> segments = new ArrayList<>();
    if (results != null) {
        for (Object resultItem : results) {
            if (resultItem instanceof HashMap) {
                HashMap<String, Object> segmentMap = (HashMap<String, Object>) resultItem;

                Float xFloat = (Float) segmentMap.get("x");
                Float yFloat = (Float) segmentMap.get("y");
                Float widthFloat = (Float) segmentMap.get("width");
                Float heightFloat = (Float) segmentMap.get("height");
                Float confidenceFloat = (Float) segmentMap.get("confidence");
                Integer indexInt = (Integer) segmentMap.get("class"); // Assuming "class" key for index
                String label = indexInt != null && indexInt < predictor.labels.size()
                        ? predictor.labels.get(indexInt) : "";
                List<List<HashMap<String, Integer>>> polygonsList = (List<List<HashMap<String, Integer>>>) segmentMap.get("polygons");

                if (xFloat != null && yFloat != null && widthFloat != null && heightFloat != null &&
                        confidenceFloat != null && indexInt != null && polygonsList != null) {

                    Map<String, Object> processedSegmentMap = new HashMap<>();
                    processedSegmentMap.put("x", xFloat);
                    processedSegmentMap.put("y", yFloat);
                    processedSegmentMap.put("width", widthFloat);
                    processedSegmentMap.put("height", heightFloat);
                    processedSegmentMap.put("confidence", confidenceFloat);
                    processedSegmentMap.put("index", indexInt);
                    processedSegmentMap.put("label", label);

                    List<List<Map<String, Integer>>> processedPolygons = new ArrayList<>();
                    for (List<HashMap<String, Integer>> polygon : polygonsList) {
                        List<Map<String, Integer>> processedPolygon = new ArrayList<>();
                        for (HashMap<String, Integer> point : polygon) {
                            processedPolygon.add(new HashMap<>(point)); // Create a new map for immutability
                        }
                        processedPolygons.add(processedPolygon);
                    }
                    processedSegmentMap.put("polygons", processedPolygons);

                    segments.add(processedSegmentMap);
                } else {
                    
                }
            } else {
               
            }
        }
    }
    resultStreamHandler.sink(segments);
});
        }

        predictor.setFpsRateCallback(fpsRateStreamHandler::sink);
        predictor.setInferenceTimeCallback(inferenceTimeStreamHandler::sink);
    }

    private void setConfidenceThreshold(MethodCall call, MethodChannel.Result result) {
        Object confidenceObject = call.argument("confidence");
        if (confidenceObject != null) {
            final double confidence = (double) confidenceObject;
            predictor.setConfidenceThreshold((float) confidence);
        }
    }

    private void setIouThreshold(MethodCall call, MethodChannel.Result result) {
        Object iouObject = call.argument("iou");
        if (iouObject != null) {
            final double iou = (double) iouObject;
            if (predictor instanceof Detector) {
                ((Detector) predictor).setIouThreshold((float) iou);
            } else if (predictor instanceof Segmenter) {
                ((Segmenter) predictor).setIouThreshold((float) iou);
            }
        }
    }

    private void setNumItemsThreshold(MethodCall call, MethodChannel.Result result) {
        Object numItemsObject = call.argument("numItems");
        if (numItemsObject != null) {
            final int numItems = (int) numItemsObject;
            if (predictor instanceof Detector) {
                ((Detector) predictor).setNumItemsThreshold(numItems);
            } else if (predictor instanceof Segmenter) {
                ((Segmenter) predictor).setNumItemsThreshold(numItems);
            }
        }
    }

    private void setLensDirection(MethodCall call, MethodChannel.Result result) {
        Object directionObject = call.argument("direction");
        if (directionObject != null) {
            final int direction = (int) directionObject;
            cameraPreview.setCameraFacing(direction);
        }
    }

    private void closeCamera(MethodCall call, MethodChannel.Result result) {
        // TODO: Implement close camera logic if needed
    }

    private void startCamera(MethodCall call, MethodChannel.Result result) {
        // TODO: Implement start camera logic if needed
    }

    private void pauseLivePrediction(MethodCall call, MethodChannel.Result result) {
        // TODO: Implement pause live prediction logic if needed
    }

    private void resumeLivePrediction(MethodCall call, MethodChannel.Result result) {
        // TODO: Implement resume live prediction logic if needed
    }

    private void detectImage(MethodCall call, MethodChannel.Result result) {
        if (predictor != null && predictor instanceof Detector) {
            Object imagePathObject = call.argument("imagePath");
            if (imagePathObject != null) {
                final String imagePath = (String) imagePathObject;
                Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
                final float[][] res = (float[][]) predictor.predict(bitmap);

                float scaleFactor = widthDp / bitmap.getWidth();
                float newHeight = bitmap.getHeight() * scaleFactor;
                List<Map<String, Object>> objects = new ArrayList<>();
                for (float[] obj : res) {
                    Map<String, Object> objectMap = new HashMap<>();

                    float x = obj[0] * widthDp;
                    float y = obj[1] * newHeight;
                    float width = obj[2] * widthDp;
                    float height = obj[3] * newHeight;
                    float confidence = obj[4];
                    int index = (int) obj[5];
                    String label = index < predictor.labels.size() ? predictor.labels.get(index) : "";

                    objectMap.put("x", x);
                    objectMap.put("y", y);
                    objectMap.put("width", width);
                    objectMap.put("height", height);
                    objectMap.put("confidence", confidence);
                    objectMap.put("index", index);
                    objectMap.put("label", label);

                    objects.add(objectMap);
                }
                result.success(objects);
            }
        } else {
            result.error("PredictorError", "Predictor is not initialized for object detection", null);
        }
    }

private void segmentImage(MethodCall call, MethodChannel.Result result) {
    if (predictor != null && predictor instanceof Segmenter) {
        Object imagePathObject = call.argument("imagePath");
        if (imagePathObject != null) {
            final String imagePath = (String) imagePathObject;
            Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
            final Object prediction = predictor.predict(bitmap);

           
if (prediction instanceof Object[]) {
    Object[] array = (Object[]) prediction;
 
}
            if (prediction instanceof Object[]) {
                Object[] results = (Object[]) prediction;
                List<Map<String, Object>> segments = new ArrayList<>();

                for (Object detectionObj : results) {
                   
                    if (detectionObj instanceof Map) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> detection = (Map<String, Object>) detectionObj;
                         Integer indexInt = (Integer) detection.get("class"); // Assuming "class" key for index
                String label = indexInt != null && indexInt < predictor.labels.size()
                        ? predictor.labels.get(indexInt) : "";
                        detection.put("label", label);
                        segments.add(detection);
                    } else {
                        Log.e(TAG, "Unexpected detection object type inside list: " + detectionObj.getClass());
                    }
                }

                result.success(segments);
            } else if (prediction instanceof Map) {
                // Sometimes prediction might be a single HashMap instead of List
                @SuppressWarnings("unchecked")
                Map<String, Object> detection = (Map<String, Object>) prediction;
                List<Map<String, Object>> segments = new ArrayList<>();
                segments.add(detection);
                result.success(segments);
            } else {
                result.error("PredictorError", "Unexpected prediction result type: " + prediction.getClass(), null);
            }
        }
    } else {
        result.error("PredictorError", "Predictor is not initialized for segmentation", null);
    }
}



    private void classifyImage(MethodCall call, MethodChannel.Result result) {
        if (predictor != null) {
            Object imagePathObject = call.argument("imagePath");
            if (imagePathObject != null) {
                final String imagePath = (String) imagePathObject;
                Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
                final List<ClassificationResult> res = (List<ClassificationResult>) predictor.predict(bitmap);

                List<Map<String, Object>> objects = new ArrayList<>();
                for (ClassificationResult classificationResult : res) {
                    Map<String, Object> objectMap = new HashMap<>();

                    objectMap.put("confidence", classificationResult.confidence);
                    objectMap.put("index", classificationResult.index);
                    objectMap.put("label", classificationResult.label);
                    objects.add(objectMap);
                }

                result.success(objects);
            }
        }
    }


    private void setScaleFactor(MethodCall call, MethodChannel.Result result) {
        Object factorObject = call.argument("ratio");
        if (factorObject != null) {
            final double factor = (double) factorObject;
            cameraPreview.setScaleFactor(factor);
        }
    }
}
