// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.*
import android.util.Log
import kotlin.math.max
import kotlin.math.min

class PoseEstimator(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private var confidenceThreshold: Float = 0.25f,   // Can be changed as needed
    private var iouThreshold: Float = 0.7f,
    private var numItemsThreshold: Int = 30
) : BasePredictor() {

    private enum class OutputLayout {
        FEATURES_FIRST,
        ANCHORS_FIRST
    }

    companion object {
        // xywh(4) + conf(1) + keypoints(17*3=51) = 56
        private const val OUTPUT_FEATURES = 56
        private const val KEYPOINTS_COUNT = 17
    }

    // Reusable float input for the CompiledModel input buffer.
    private lateinit var floatInput: FloatArray
    private lateinit var inputBitmap: Bitmap
    private lateinit var intValues: IntArray
    
    // Reuse output arrays to reduce allocations
    private var outDim2 = 0
    
    // Output dimensions
    private var batchSize = 0
    private var numAnchors = 0
    private var isEndToEnd = false
    private lateinit var outputLayout: OutputLayout

    init {
        val loadedLabels = YOLOFileUtils.loadModelLabels(context, modelPath)
        if (loadedLabels != null) {
            this.labels = loadedLabels
            Log.i("PoseEstimator", "Labels successfully loaded from appended ZIP.")
        } else if (this.labels.isEmpty()) {
            Log.w("PoseEstimator", "No embedded labels found and none provided; detections may lack class names.")
        }

        rtModel = InferenceModel.create(context, modelPath, useGpu, "PoseEstimator")

        val inDims = rtModel.inputDims
        val inHeight = if (inDims.size >= 4) inDims[1] else 640
        val inWidth = if (inDims.size >= 4) inDims[2] else 640
        inputSize = com.ultralytics.yolo.Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)

        val outputShape = rtModel.outputDims.getOrNull(0) ?: IntArray(0)
        batchSize = if (outputShape.isNotEmpty()) outputShape[0] else 1
        isEndToEnd = outputShape[2] < outputShape[1] && outputShape[2] >= 6
        outputLayout = when {
            outputShape[1] == OUTPUT_FEATURES -> OutputLayout.FEATURES_FIRST
            outputShape[2] == OUTPUT_FEATURES || isEndToEnd -> OutputLayout.ANCHORS_FIRST
            else -> throw IllegalArgumentException(
                "Unexpected output feature size. Expected $OUTPUT_FEATURES or end-to-end rows, Actual=${outputShape.contentToString()}"
            )
        }

        val outFeatures: Int
        when (outputLayout) {
            OutputLayout.FEATURES_FIRST -> {
                outFeatures = outputShape[1]
                numAnchors = outputShape[2]
            }
            OutputLayout.ANCHORS_FIRST -> {
                outFeatures = outputShape[2]
                numAnchors = outputShape[1]
            }
        }

        outDim2 = outputShape[2]

        floatInput = FloatArray(inWidth * inHeight * 3)
        inputBitmap = Bitmap.createBitmap(inWidth, inHeight, Bitmap.Config.ARGB_8888)
        intValues = IntArray(inWidth * inHeight)
    }

    override fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean, isLandscape: Boolean): YOLOResult {
        t0 = System.nanoTime()
        ImageUtils.prepareBitmapForModel(
            bitmap = bitmap,
            targetBitmap = inputBitmap,
            rotateForCamera = rotateForCamera,
            isLandscape = isLandscape,
            isFrontCamera = isFrontCamera,
            rotationDegrees = cameraRotationDegrees
        )
        ImageUtils.copyRgbBitmapToFloatArray(
            inputBitmap,
            floatInput,
            intValues,
            channelsFirst = rtModel.inputUsesNchw
        )

        val preEnd = System.nanoTime()
        val flat = rtModel.run(floatInput)[0]
        val inferEnd = System.nanoTime()
        // Decode straight from the flat output - featureValue() handles both layouts without a reshape copy.
        val rawDetections = postProcessPose(
            flat = flat,
            numAnchors = numAnchors,
            confidenceThreshold = confidenceThreshold,
            iouThreshold = iouThreshold,
            origWidth = origWidth,
            origHeight = origHeight
        )

        // Apply numItemsThreshold limit
        val limitedDetections = rawDetections.take(numItemsThreshold)
        
        val boxes = limitedDetections.map { it.box }
        val keypointsList = limitedDetections.map { it.keypoints }

        val timing = finishTiming(preEnd, inferEnd)
        // Pack into YOLOResult and return
        return YOLOResult(
            origShape = com.ultralytics.yolo.Size(origWidth, origHeight),
            boxes = boxes,
            keypointsList = keypointsList,
            speed = timing.speedMs,
            fps = timing.fps,
            preMs = timing.preMs,
            inferenceMs = timing.inferenceMs,
            postMs = timing.postMs,
            names = labels
        )
    }

    private fun postProcessPose(
        flat: FloatArray,
        numAnchors: Int,
        confidenceThreshold: Float,
        iouThreshold: Float,
        origWidth: Int,
        origHeight: Int
    ): List<PoseDetection> {
        if (isEndToEnd) {
            return postProcessEndToEndPose(flat, confidenceThreshold, origWidth, origHeight)
        }

        val detections = mutableListOf<PoseDetection>()

        for (j in 0 until numAnchors) {
            val rawX = featureValue(flat, 0, j)
            val rawY = featureValue(flat, 1, j)
            val rawW = featureValue(flat, 2, j)
            val rawH = featureValue(flat, 3, j)
            val conf = featureValue(flat, 4, j)

            if (conf < confidenceThreshold) continue

            val outputRect = RectF(
                rawX - rawW / 2f,
                rawY - rawH / 2f,
                rawX + rawW / 2f,
                rawY + rawH / 2f
            )
            val outputIsNormalized = outputCoordinatesAreNormalized(outputRect)
            val rectF = inputRectFromOutputRect(
                outputRect,
                origWidth,
                origHeight,
                outputIsNormalized
            ) ?: continue
            val normBox = normalizedRectFromInputRect(rectF, origWidth, origHeight)

            val kpArray = mutableListOf<Pair<Float, Float>>()
            val kpConfArray = mutableListOf<Float>()
            for (k in 0 until KEYPOINTS_COUNT) {
                val rawKx = featureValue(flat, 5 + k * 3, j)
                val rawKy = featureValue(flat, 5 + k * 3 + 1, j)
                val kpC   = featureValue(flat, 5 + k * 3 + 2, j)

                val (finalKx, finalKy) = inputPointFromOutputPoint(
                    rawKx,
                    rawKy,
                    origWidth,
                    origHeight,
                    outputIsNormalized
                )

                kpArray.add(finalKx to finalKy)
                kpConfArray.add(kpC)
            }

            val xynList = kpArray.map { (fx, fy) ->
                (fx / origWidth) to (fy / origHeight)
            }
            val boxObj = Box(0, labelName(0), conf, rectF, normBox)
            
            val keypoints = Keypoints(
                xyn = xynList,
                xy = kpArray,
                conf = kpConfArray
            )
            
            detections.add(
                PoseDetection(
                    box = boxObj,
                    keypoints = keypoints
                )
            )
        }

        val finalDetections = nmsPoseDetections(detections, iouThreshold)
        return finalDetections
    }

    private fun postProcessEndToEndPose(
        flat: FloatArray,
        confidenceThreshold: Float,
        origWidth: Int,
        origHeight: Int
    ): List<PoseDetection> {
        val detections = mutableListOf<PoseDetection>()
        val fieldCount = outDim2
        val keypointStart = if ((fieldCount - 6) % 3 == 0) 6 else 5
        val keypointCount = (fieldCount - keypointStart) / 3

        for (j in 0 until numAnchors) {
            val conf = featureValue(flat, 4, j)
            if (conf < confidenceThreshold) continue

            val outputRect = RectF(
                featureValue(flat, 0, j),
                featureValue(flat, 1, j),
                featureValue(flat, 2, j),
                featureValue(flat, 3, j)
            )
            val outputIsNormalized = outputCoordinatesAreNormalized(outputRect)
            val rectF = inputRectFromOutputRect(
                outputRect,
                origWidth,
                origHeight,
                outputIsNormalized
            ) ?: continue
            val normBox = normalizedRectFromInputRect(rectF, origWidth, origHeight)

            val kpArray = mutableListOf<Pair<Float, Float>>()
            val kpConfArray = mutableListOf<Float>()
            for (k in 0 until keypointCount) {
                val rawKx = featureValue(flat, keypointStart + k * 3, j)
                val rawKy = featureValue(flat, keypointStart + k * 3 + 1, j)
                val kpC = featureValue(flat, keypointStart + k * 3 + 2, j)
                kpArray.add(
                    inputPointFromOutputPoint(
                        rawKx,
                        rawKy,
                        origWidth,
                        origHeight,
                        outputIsNormalized
                    )
                )
                kpConfArray.add(kpC)
            }

            detections.add(
                PoseDetection(
                    box = Box(0, labelName(0), conf, rectF, normBox),
                    keypoints = Keypoints(
                        xyn = kpArray.map { (fx, fy) -> (fx / origWidth) to (fy / origHeight) },
                        xy = kpArray,
                        conf = kpConfArray
                    )
                )
            )
        }
        return detections
    }

    private fun featureValue(
        flat: FloatArray,
        featureIndex: Int,
        anchorIndex: Int
    ): Float {
        // Row stride is outDim2 for both layouts of the flat [1, d1, d2] output
        return when (outputLayout) {
            OutputLayout.FEATURES_FIRST -> flat[featureIndex * outDim2 + anchorIndex]
            OutputLayout.ANCHORS_FIRST -> flat[anchorIndex * outDim2 + featureIndex]
        }
    }


    private fun nmsPoseDetections(
        detections: List<PoseDetection>,
        iouThreshold: Float
    ): List<PoseDetection> {
        val confidenceThreshold = 0.25f  // Hardcoded second-pass threshold
        val filteredDetections = detections.filter { it.box.conf >= confidenceThreshold }
        
        if (filteredDetections.size <= 1) {
            return filteredDetections
        }
        
        val sorted = filteredDetections.sortedByDescending { it.box.conf }
        val picked = mutableListOf<PoseDetection>()
        val used = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (used[i]) continue

            val d1 = sorted[i]
            picked.add(d1)

            for (j in i + 1 until sorted.size) {
                if (used[j]) continue
                val d2 = sorted[j]
                if (iou(d1.box.xywh, d2.box.xywh) > iouThreshold) {
                    used[j] = true
                }
            }
        }
        return picked
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val interW = max(0f, interRight - interLeft)
        val interH = max(0f, interBottom - interTop)
        val interArea = interW * interH
        val unionArea = a.width() * a.height() + b.width() * b.height() - interArea
        return if (unionArea <= 0f) 0f else (interArea / unionArea)
    }


    override fun setConfidenceThreshold(conf: Double) {
        confidenceThreshold = conf.toFloat()
        super.setConfidenceThreshold(conf)
    }

    override fun setIouThreshold(iou: Double) {
        iouThreshold = iou.toFloat()
        super.setIouThreshold(iou)
    }
    
    override fun setNumItemsThreshold(n: Int) {
        numItemsThreshold = n
        super.setNumItemsThreshold(n)
    }

    override fun getConfidenceThreshold(): Double {
        return confidenceThreshold.toDouble()
    }

    override fun getIouThreshold(): Double {
        return iouThreshold.toDouble()
    }

    private data class PoseDetection(
        val box: Box,
        val keypoints: Keypoints
    )
    
}
