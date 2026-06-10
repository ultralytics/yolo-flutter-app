// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log

/**
 * High-performance ObjectDetector on LiteRT 2.x ([LiteRtModel]).
 * - Letterbox -> getPixels -> FloatArray with reusable buffers
 * - Reuses Bitmap / FloatArray / output arrays to reduce allocations
 */
class ObjectDetector(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private var numItemsThreshold: Int = 30
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

    // (3) Reusable float input (1 * height * width * 3) for the CompiledModel input buffer.
    private lateinit var floatInput: FloatArray

    init {
        // Labels from the model metadata: Ultralytics' appended ZIP, falling back to standard embedded TFLite
        // (FlatBuffers) metadata so drag-and-dropped custom models keep their labels.
        val loadedLabels = YOLOFileUtils.loadModelLabels(context, modelPath)
        if (loadedLabels != null) {
            this.labels = loadedLabels
            Log.i(TAG, "Labels loaded from model metadata.")
        } else if (this.labels.isEmpty()) {
            Log.w(TAG, "No embedded labels found and none provided; detections may lack class names.")
        }

        rtModel = InferenceModel.create(context, modelPath, useGpu, "ObjectDetector")

        // Input dims [1, H, W, 3].
        val inDims = rtModel.inputDims
        val inHeight = if (inDims.size >= 4) inDims[1] else 640
        val inWidth = if (inDims.size >= 4) inDims[2] else 640
        inputSize = Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)

        // Output dims [1, features, anchors], e.g. [1, 84, 8400]. Fall back to deriving from the element count.
        val outDims = rtModel.outputDims.getOrNull(0) ?: IntArray(0)
        if (outDims.size >= 3) {
            out1 = outDims[1]
            out2 = outDims[2]
        } else {
            val count = rtModel.outputElementCounts.getOrElse(0) { 0 }
            val features = labels.size + 4
            if (features in 1..count && count % features == 0) {
                out1 = features
                out2 = count / features
            }
        }

        initPreprocessingResources(inWidth, inHeight)
        rawOutput = Array(1) { Array(out1) { FloatArray(out2) } }
    }

    private fun initPreprocessingResources(width: Int, height: Int) {
        scaledBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        intValues = IntArray(width * height)
        floatInput = FloatArray(width * height * 3)
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
        ImageUtils.copyRgbBitmapToFloatArray(
            scaledBitmap,
            floatInput,
            intValues,
            INPUT_MEAN,
            INPUT_STANDARD_DEVIATION
        )

        // ======== Inference ============
        val preEnd = System.nanoTime()
        val outputs = rtModel.run(floatInput)
        val inferEnd = System.nanoTime()

        // Reshape the flat [out1*out2] output back into rawOutput[0][out1][out2] for the existing postprocess.
        val flat = outputs[0]
        val rows = rawOutput[0]
        var idx = 0
        for (i in 0 until out1) {
            val row = rows[i]
            for (j in 0 until out2) {
                row[j] = flat[idx++]
            }
        }

        // ======== Post-processing (same as existing code) ============
        val outHeight = rawOutput[0].size      // out1
        val outWidth = rawOutput[0][0].size      // out2

        val resultBoxes = if (outWidth < outHeight && outWidth >= 6) {
            postprocessEndToEnd(rawOutput[0])
        } else {
            val classCount = (outHeight - 4).coerceAtLeast(0)
            postprocess(
                rawOutput[0],
                w = outWidth,   // width is out2
                h = outHeight,  // height is out1
                confidenceThreshold = confidenceThreshold,
                iouThreshold = iouThreshold,
                numItemsThreshold = numItemsThreshold,
                numClasses = classCount
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
                    boxes.add(Box(classIdx, labelName(classIdx), boxArray[4], rect, normRect))
                }
            }
        }

        val timing = finishTiming(preEnd, inferEnd)

        return YOLOResult(
            origShape = com.ultralytics.yolo.Size(origWidth, origHeight),
            boxes = boxes,
            speed = timing.speedMs,
            fps = timing.fps,
            preMs = timing.preMs,
            inferenceMs = timing.inferenceMs,
            postMs = timing.postMs,
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
