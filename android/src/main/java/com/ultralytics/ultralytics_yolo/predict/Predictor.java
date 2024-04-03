package com.ultralytics.ultralytics_yolo.predict;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import androidx.annotation.Keep;
import androidx.camera.core.ImageProxy;

import com.ultralytics.ultralytics_yolo.models.YoloModel;

import org.yaml.snakeyaml.Yaml;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Map;

public abstract class Predictor {
public static final int INPUT_SIZE = 320;
        protected final Context context;
    public final ArrayList<String> labels = new ArrayList<>();

    static {
        System.loadLibrary("ultralytics");
    }

    protected Predictor(Context context) {
        this.context = context;
    }

    public abstract void loadModel(YoloModel yoloModel, boolean useGpu) throws Exception;

    protected void loadLabels(AssetManager assetManager, String metadataPath) throws IOException {
        InputStream inputStream;
        Yaml yaml = new Yaml();

        // Local metadata file from Flutter project
        if (metadataPath.startsWith("flutter_assets")) {
            inputStream = assetManager.open(metadataPath);
        }
        // Absolute path
        else {
            inputStream = Files.newInputStream(Paths.get(metadataPath));
        }

        Map<String, Object> data = yaml.load(inputStream);
        Map<Integer, String> names = ((Map<Integer, String>) data.get("names"));

        labels.clear();
        labels.addAll(names.values());

        inputStream.close();
    }

    public abstract Object predict(Bitmap bitmap);

    public abstract void predict(ImageProxy imageProxy, boolean isMirrored);

    public abstract void setConfidenceThreshold(float confidence);

    public abstract void setInferenceTimeCallback(FloatResultCallback callback);

    public abstract void setFpsRateCallback(FloatResultCallback callback);

    public interface FloatResultCallback {
        @Keep()
        void onResult(float result);
    }
}
