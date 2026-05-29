// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.yaml.snakeyaml.Yaml
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
/**
 * High-performance ObjectDetector.
 * - Performs "letterbox -> getPixels -> ByteBuffer" with reusable buffers
 * - Reuses Bitmap / ByteBuffer to reduce allocations
 * - Reuses inference output arrays
 */
class ObjectDetector(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private var numItemsThreshold: Int = 30,
    private val customOptions: Interpreter.Options? = null
) : BasePredictor() {
    // Inference output dimensions
    private var out1 = 0
    private var out2 = 0
    // Reuse inference output array ([1][out1][out2])
    private lateinit var rawOutput: Array<Array<FloatArray>>

    // ======== Workspace for fast preprocessing ========
    // (1) Temporary letterboxed Bitmap matching model input size
    private lateinit var scaledBitmap: Bitmap

    // (2) Array to temporarily store pixels (inWidth*inHeight)
    private lateinit var intValues: IntArray

    // (3) ByteBuffer for TFLite input (1 * height * width * 3 * 4 bytes)
    private lateinit var inputBuffer: ByteBuffer

    // CPU interpreter options. The GPU delegate is NOT added here; createInterpreterFastestFirst owns delegate
    // selection (GPU first, closing it on failure) and falls back to these options for the XNNPACK CPU path.
    private val interpreterOptions: Interpreter.Options = (customOptions ?: Interpreter.Options()).apply {
        if (customOptions == null) {
            setNumThreads(Runtime.getRuntime().availableProcessors())
        }
    }

    // ========== TFLite Interpreter ==========
    // Use protected var interpreter: Interpreter? = null from BasePredictor if available
    // Otherwise, keep it in this class as usual
    init {
        val modelBuffer  = YOLOUtils.loadModelFile(context, modelPath)

        /* --- Get labels from metadata (try Appended ZIP → FlatBuffers in order) --- */
        val loadedLabels = YOLOFileUtils.loadLabelsFromAppendedZip(context, modelPath)
        var labelsWereLoaded = loadedLabels != null

        if (loadedLabels != null) {
            this.labels = loadedLabels // Use labels from appended ZIP
            Log.i(TAG, "Labels successfully loaded from appended ZIP.")
        } else {
            Log.w(TAG, "Could not load labels from appended ZIP, trying FlatBuffers metadata...")
            // Try FlatBuffers as a fallback
            if (loadLabelsFromFlatbuffers(modelBuffer)) {
                labelsWereLoaded = true
                Log.i(TAG, "Labels successfully loaded from FlatBuffers metadata.")
            }
        }

        if (!labelsWereLoaded) {
            Log.w(TAG, "No embedded labels found from appended ZIP or FlatBuffers. Using labels passed via constructor (if any) or an empty list.")
            if (this.labels.isEmpty()) {
                 Log.w(TAG, "Warning: No labels loaded and no labels provided via constructor. Detections might lack class names.")
            }
        }

        interpreter = createInterpreterFastestFirst(modelBuffer, useGpu, interpreterOptions, "ObjectDetector")
        // Call allocateTensors() once during initialization, not in the inference loop
        interpreter.allocateTensors()

        // Check input shape (example: [1, inHeight, inWidth, 3])
        val inputShape = interpreter.getInputTensor(0).shape()
        val inBatch = inputShape[0]         // Usually 1
        val inHeight = inputShape[1]        // Example: 320
        val inWidth = inputShape[2]         // Example: 320
        val inChannels = inputShape[3]      // 3 (RGB)
        require(inBatch == 1 && inChannels == 3) {
            "Input tensor shape not supported. Expected [1, H, W, 3]. But got ${inputShape.joinToString()}"
        }
        inputSize = Size(inWidth, inHeight) // Set variable in BasePredictor
        modelInputSize = Pair(inWidth, inHeight)

        // Output shape (varies by model, modify as needed)
        // Example: [1, 84, 2100] = [batch, outHeight, outWidth]
        val outputShape = interpreter.getOutputTensor(0).shape()
        out1 = outputShape[1] // 84
        out2 = outputShape[2] // 2100

        // Allocate preprocessing resources
        initPreprocessingResources(inWidth, inHeight)

        // Allocate inference output arrays
        rawOutput = Array(1) { Array(out1) { FloatArray(out2) } }
    }

    /* =================================================================== */
    /*                 metadata helper functions (Kotlin)                 */
    /* =================================================================== */

    /**
     * ────────────────────────────────────────────────────────────────
     *  Load labels from FlatBuffers (metadata.yaml) - based on old code
     *  - Scan all associatedFileNames
     *  - Parse YAML as Map<Int,String>
     *  - Use values directly as List and assign to labels
     * ────────────────────────────────────────────────────────────────
     */
    private fun loadLabelsFromFlatbuffers(buf: MappedByteBuffer): Boolean = try {
        val extractor = MetadataExtractor(buf)
        val files = extractor.associatedFileNames
        if (!files.isNullOrEmpty()) {
            for (fileName in files) {
                extractor.getAssociatedFile(fileName)?.use { stream ->
                    val fileString = String(stream.readBytes(), Charsets.UTF_8)

                    val yaml = Yaml()
                    @Suppress("UNCHECKED_CAST")
                    val data = yaml.load<Map<String, Any>>(fileString)
                    if (data != null && data.containsKey("names")) {
                        val namesMap = data["names"] as? Map<Int, String>
                        if (namesMap != null) {
                            labels = namesMap.values.toList()          // Same as old code
                            return true
                        }
                    }
                }
            }
        }
        false
    } catch (e: Exception) {
        Log.e(TAG, "Failed to extract metadata: ${e.message}")
        false
    }


    private fun initPreprocessingResources(width: Int, height: Int) {
        // ARGB_8888 Bitmap for input size (e.g., 320x320)
        scaledBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)

        // Int array for pixel reading
        intValues = IntArray(width * height)

        // Buffer for TFLite input
        inputBuffer = ByteBuffer.allocateDirect(1 * width * height * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }
    }

    /**
     * Main inference method
     * - Preprocessing: rotate if needed, letterbox into scaledBitmap, then getPixels → inputBuffer
     * - TFLite run
     * - Postprocessing (NMS via JNI, etc.)
     * @param bitmap Input bitmap to process
     * @param origWidth Original width of the source image
     * @param origHeight Original height of the source image
     * @param rotateForCamera Whether this is a camera feed that requires rotation (true) or a single image (false)
     * @return YOLOResult containing detection results
     */
    override fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean, isLandscape: Boolean): YOLOResult {
        val overallStartTime = System.nanoTime()
        t0 = overallStartTime

        ImageUtils.prepareBitmapForModel(
            bitmap = bitmap,
            targetBitmap = scaledBitmap,
            rotateForCamera = rotateForCamera,
            isLandscape = isLandscape,
            isFrontCamera = isFrontCamera,
            rotationDegrees = cameraRotationDegrees
        )
        ImageUtils.copyRgbBitmapToFloatBuffer(
            scaledBitmap,
            inputBuffer,
            intValues,
            INPUT_MEAN,
            INPUT_STANDARD_DEVIATION
        )

        // ======== Inference ============
        interpreter.run(inputBuffer, rawOutput)

        // ======== Post-processing (same as existing code) ============
        val outHeight = rawOutput[0].size      // out1
        val outWidth = rawOutput[0][0].size      // out2

        val resultBoxes = if (outWidth < outHeight && outWidth >= 6) {
            postprocessEndToEnd(rawOutput[0])
        } else {
            postprocess(
                rawOutput[0],
                w = outWidth,   // width is out2
                h = outHeight,  // height is out1
                confidenceThreshold = confidenceThreshold,
                iouThreshold = iouThreshold,
                numItemsThreshold = numItemsThreshold,
                numClasses = labels.size
            )
        }
        // Convert to Box list
        val boxes = mutableListOf<Box>()
        for (boxArray in resultBoxes) {
            if (boxArray.size >= 6) {
                val rect = inputRectFromOutputRect(
                    RectF(boxArray[0], boxArray[1], boxArray[0] + boxArray[2], boxArray[1] + boxArray[3]),
                    origWidth,
                    origHeight
                )

                if (rect != null) {
                    val normRect = normalizedRectFromInputRect(rect, origWidth, origHeight)
                    val classIdx = boxArray[5].toInt()
                    val label = if (classIdx in labels.indices) labels[classIdx] else "Unknown"
                    boxes.add(Box(classIdx, label, boxArray[4], rect, normRect))
                }
            }
        }

        val totalMs = (System.nanoTime() - overallStartTime) / 1_000_000.0

        updateTiming() // This updates t0, t1, t2, t3, t4 based on its own logic

        return YOLOResult(
            origShape = com.ultralytics.yolo.Size(origWidth, origHeight),
            boxes = boxes,
            speed = totalMs, // Actual processing time in milliseconds for this frame
            fps = if (t4 > 0.0) 1.0 / t4 else 0.0, // Smoothed FPS from BasePredictor (t4 is smoothed dt)
            names = labels
        )
    }

    // Thresholds (like setConfidenceThreshold, setIouThreshold in TFLiteDetector)
    private var confidenceThreshold = 0.25f
    private var iouThreshold = 0.7f

    override fun setConfidenceThreshold(conf: Double) {
        confidenceThreshold = conf.toFloat()
        super.setConfidenceThreshold(conf)
    }

    override fun setIouThreshold(iou: Double) {
        iouThreshold = iou.toFloat()
        super.setIouThreshold(iou)
    }

    override fun getConfidenceThreshold(): Double {
        return confidenceThreshold.toDouble()
    }

    override fun getIouThreshold(): Double {
        return iouThreshold.toDouble()
    }

    override fun setNumItemsThreshold(n: Int) {
        numItemsThreshold = n
        super.setNumItemsThreshold(n)
    }

    // Post-processing via JNI
    private external fun postprocess(
        predictions: Array<FloatArray>,
        w: Int,
        h: Int,
        confidenceThreshold: Float,
        iouThreshold: Float,
        numItemsThreshold: Int,
        numClasses: Int
    ): Array<FloatArray>

    private fun postprocessEndToEnd(predictions: Array<FloatArray>): Array<FloatArray> {
        val boxes = mutableListOf<FloatArray>()
        for (row in predictions) {
            val confidence = row[4]
            if (confidence <= confidenceThreshold) continue
            val x1 = row[0]
            val y1 = row[1]
            val x2 = row[2]
            val y2 = row[3]
            boxes.add(
                floatArrayOf(
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    confidence,
                    row[5]
                )
            )
            if (boxes.size >= numItemsThreshold) break
        }
        return boxes.toTypedArray()
    }

    companion object {
        private const val TAG = "ObjectDetector"
        // Load JNI library
        init {
            System.loadLibrary("ultralytics")
        }
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
    }
}
