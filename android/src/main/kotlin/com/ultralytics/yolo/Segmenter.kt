// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.yaml.snakeyaml.Yaml
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class Segmenter(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private var numItemsThreshold: Int = 30,
    private val customOptions: Interpreter.Options? = null
) : BasePredictor() {

    private val boxFeatureLength = 4  // (x, y, w, h)
    private val maskConfidenceLength = 32
    private var numClasses = 0
    private var out0NumFeatures = 0
    private var out0NumAnchors = 0
    private var maskH = 0
    private var maskW = 0
    private var maskC = 0
    private var isEndToEnd = false

    // TFLite Interpreter options
    private val interpreterOptions = (customOptions ?: Interpreter.Options()).apply {
        // If no custom options provided, use default threads
        if (customOptions == null) {
            setNumThreads(Runtime.getRuntime().availableProcessors())
        }

        if (useGpu) {
            try {
                addDelegate(GpuDelegate())
            } catch (e: Exception) {
                Log.e("Segmenter", "GPU delegate error: ${e.message}")
            }
        }
    }

    // Reuse ByteBuffer for input to reduce allocations
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var inputBitmap: Bitmap
    private lateinit var intValues: IntArray

    // Reuse output arrays to reduce allocations
    private lateinit var output0: Array<Array<FloatArray>>
    private lateinit var output1: Array<Array<Array<FloatArray>>>

    init {

        // Load model file (automatic extension appending)
        val modelBuffer = YOLOUtils.loadModelFile(context, modelPath)

        // ===== Load label information (try Appended ZIP → FlatBuffers in order) =====
        val loadedLabels = YOLOFileUtils.loadLabelsFromAppendedZip(context, modelPath)
        var labelsWereLoaded = loadedLabels != null

        if (loadedLabels != null) {
            this.labels = loadedLabels // Use labels from appended ZIP
            Log.i("Segmenter", "Labels successfully loaded from appended ZIP.")
        } else {
            Log.w("Segmenter", "Could not load labels from appended ZIP, trying FlatBuffers metadata...")
            // Try FlatBuffers as a fallback
            if (loadLabelsFromFlatbuffers(modelBuffer)) {
                labelsWereLoaded = true
                Log.i("Segmenter", "Labels successfully loaded from FlatBuffers metadata.")
            }
        }

        if (!labelsWereLoaded) {
            Log.w(
                "Segmenter",
                "No embedded labels found from appended ZIP or FlatBuffers. Using labels passed via constructor (if any) or an empty list."
            )
            if (this.labels.isEmpty()) {
                Log.w(
                    "Segmenter",
                    "Warning: No labels loaded and no labels provided via constructor. Detections might lack class names."
                )
            }
        }

        // Create Interpreter
        interpreter = Interpreter(modelBuffer, interpreterOptions)
        // Call allocateTensors() once during initialization
        interpreter.allocateTensors()

        // Input tensor shape: [1, height, width, 3]
        val inputShape = interpreter.getInputTensor(0).shape()
        // Example: [1, 640, 640, 3]
        val inHeight = inputShape[1]
        val inWidth = inputShape[2]
        // Set variables in BasePredictor
        inputSize = com.ultralytics.yolo.Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)

        // Get and initialize output buffer sizes
        val out0Shape = interpreter.getOutputTensor(0).shape()
        val out1Shape = interpreter.getOutputTensor(1).shape()

        // Initialize output0 buffer (traditional [1,116,2100] or end-to-end [1,300,38])
        val batch0 = out0Shape[0]
        isEndToEnd = out0Shape[2] < out0Shape[1] && out0Shape[2] >= 6
        if (isEndToEnd) {
            out0NumAnchors = out0Shape[1]
            out0NumFeatures = out0Shape[2]
            output0 = Array(batch0) { Array(out0NumAnchors) { FloatArray(out0NumFeatures) } }
        } else {
            out0NumFeatures = out0Shape[1]
            out0NumAnchors = out0Shape[2]
            output0 = Array(batch0) { Array(out0NumFeatures) { FloatArray(out0NumAnchors) } }
        }

        // Initialize output1 buffer (example: [1,80,80,32])
        val batch1 = out1Shape[0]
        maskH = out1Shape[1]
        maskW = out1Shape[2]
        maskC = out1Shape[3]
        output1 = Array(batch1) { Array(maskH) { Array(maskW) { FloatArray(maskC) } } }

        // Initialize input buffer (direct allocation)
        val inputBytes = 1 * inHeight * inWidth * 3 * 4 // FLOAT32 is 4 bytes
        inputBuffer = ByteBuffer.allocateDirect(inputBytes).apply {
            order(ByteOrder.nativeOrder())
        }
        inputBitmap = Bitmap.createBitmap(inWidth, inHeight, Bitmap.Config.ARGB_8888)
        intValues = IntArray(inWidth * inHeight)
    }

    override fun setNumItemsThreshold(n: Int) {
        numItemsThreshold = n
        super.setNumItemsThreshold(n)
    }

    override fun predict(
        bitmap: Bitmap,
        origWidth: Int,
        origHeight: Int,
        rotateForCamera: Boolean,
        isLandscape: Boolean
    ): YOLOResult {
        t0 = System.nanoTime()

        ImageUtils.prepareBitmapForModel(
            bitmap = bitmap,
            targetBitmap = inputBitmap,
            rotateForCamera = rotateForCamera,
            isLandscape = isLandscape,
            isFrontCamera = isFrontCamera,
            rotationDegrees = cameraRotationDegrees
        )
        ImageUtils.copyRgbBitmapToFloatBuffer(inputBitmap, inputBuffer, intValues)

        // (2) Output buffer already allocated during initialization
        numClasses = out0NumFeatures - boxFeatureLength - maskConfidenceLength

        // (3) Execute inference
        val outputMap = mapOf(0 to output0, 1 to output1)
        try {
            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
        } catch (e: Exception) {
            Log.e("Segmenter", "Inference error: ${e.message}")
            val fpsDouble: Double = if (t4 > 0f) (1f / t4).toDouble() else 0.0
            return YOLOResult(
                origShape = com.ultralytics.yolo.Size(origWidth, origHeight),
                boxes = emptyList(),
                speed = t2,
                fps = fpsDouble,
                names = labels
            )
        }
        updateTiming()

        // (4) Post-processing (box + mask)
        val rawDetections = postProcessSegment(
            feature = output0[0],
            numAnchors = out0NumAnchors,
            confidenceThreshold = CONFIDENCE_THRESHOLD,
            iouThreshold = IOU_THRESHOLD
        )

        // Apply numItemsThreshold limit
        val limitedDetections = rawDetections.take(numItemsThreshold)

        val boxes = mutableListOf<Box>()
        val maskDetections = mutableListOf<Detection>()
        for (detection in limitedDetections) {
            val (boxRect, cls, score, _) = detection
            val rectF = inputRectFromOutputRect(boxRect, origWidth, origHeight) ?: continue
            val normRect = normalizedRectFromInputRect(rectF, origWidth, origHeight)
            val label = labels.getOrElse(cls) { "Unknown" }
            boxes.add(Box(cls, label, score, rectF, normRect))
            maskDetections.add(detection)
        }

        val (combinedMask, probMasks) = generateCombinedMaskImage(
            detections = maskDetections,
            protos = output1[0],
            maskW = maskW,
            maskH = maskH,
            origWidth = origWidth,
            origHeight = origHeight,
            threshold = 0.5f
        )
        val masks = Masks(probMasks ?: emptyList(), combinedMask)
        val fpsDouble: Double = if (t4 > 0f) (1f / t4).toDouble() else 0.0
        return YOLOResult(
            origShape = com.ultralytics.yolo.Size(origWidth, origHeight),
            boxes = boxes,
            masks = masks,
            speed = t2,
            fps = fpsDouble,
            names = labels
        )
    }

    private fun postProcessSegment(
        feature: Array<FloatArray>,
        numAnchors: Int,
        confidenceThreshold: Float,
        iouThreshold: Float
    ): List<Detection> {
        if (isEndToEnd) {
            return postProcessEndToEndSegment(feature, confidenceThreshold)
        }

        // Add performance measurement
        val startTime = android.os.SystemClock.elapsedRealtimeNanos()

        // Estimated capacity for results list to reduce reallocations
        val estimatedCapacity = (numAnchors * 0.05).toInt() // Assume ~5% will pass threshold
        val results = ArrayList<Detection>(estimatedCapacity)

        // Apply early filtering - Optimization: early pruning strategy
        val earlyThreshold = confidenceThreshold * 0.8f // Slightly lower threshold for first pass

        for (j in 0 until numAnchors) {
            // Check all classes instead of just first 3 to avoid bias
            var quickMaxScore = 0f
            for (c in 0 until numClasses) {
                quickMaxScore = max(quickMaxScore, feature[4 + c][j])
            }

            // Skip further processing if clearly below threshold
            if (quickMaxScore < earlyThreshold) continue

            // Continue with full processing for potential detections
            val cx = feature[0][j]
            val cy = feature[1][j]
            val w = feature[2][j]
            val h = feature[3][j]
            var maxScore = 0f
            var maxClassIdx = 0

            for (c in 0 until numClasses) {
                val score = feature[4 + c][j]
                if (score > maxScore) {
                    maxScore = score
                    maxClassIdx = c
                }
            }

            if (maxScore >= confidenceThreshold) {
                val maskCoeffs = FloatArray(maskConfidenceLength)
                val base = 4 + numClasses
                for (m in 0 until maskConfidenceLength) {
                    maskCoeffs[m] = feature[base + m][j]
                }
                val left = cx - w / 2f
                val top = cy - h / 2f
                val right = cx + w / 2f
                val bottom = cy + h / 2f
                results.add(Detection(RectF(left, top, right, bottom), maxClassIdx, maxScore, maskCoeffs))
            }
        }
        val finalDetections = mutableListOf<Detection>()
        for (classIndex in 0 until numClasses) {
            val sameClass = results.filter { it.cls == classIndex }.sortedByDescending { it.score }
            val picked = mutableListOf<Detection>()
            val used = BooleanArray(sameClass.size)
            for (i in sameClass.indices) {
                if (used[i]) continue
                val a = sameClass[i]
                picked.add(a)
                for (j in i + 1 until sameClass.size) {
                    if (used[j]) continue
                    val b = sameClass[j]
                    if (iou(a.box, b.box) > iouThreshold) {
                        used[j] = true
                    }
                }
            }
            finalDetections.addAll(picked)
        }
        return finalDetections
    }

    private fun postProcessEndToEndSegment(
        feature: Array<FloatArray>,
        confidenceThreshold: Float
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        val fieldCount = if (feature.isNotEmpty()) feature[0].size else 0
        val maskStart = if (fieldCount > 5) 6 else 5

        for (j in 0 until out0NumAnchors) {
            val confidence = feature[j][4]
            if (confidence < confidenceThreshold) continue

            val maskCoeffs = FloatArray(maskConfidenceLength)
            for (m in 0 until min(maskConfidenceLength, fieldCount - maskStart)) {
                maskCoeffs[m] = feature[j][maskStart + m]
            }
            detections.add(
                Detection(
                    RectF(feature[j][0], feature[j][1], feature[j][2], feature[j][3]),
                    if (fieldCount > 5) feature[j][5].toInt() else 0,
                    confidence,
                    maskCoeffs
                )
            )
        }
        return detections
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
        return if (unionArea <= 0f) 0f else interArea / unionArea
    }

    private fun generateCombinedMaskImage(
        detections: List<Detection>,
        protos: Array<Array<FloatArray>>,
        maskW: Int,
        maskH: Int,
        origWidth: Int,
        origHeight: Int,
        threshold: Float
    ): Pair<Bitmap?, List<List<List<Float>>>?> {
        if (detections.isEmpty()) return Pair(null, null)
        val contentRect = maskContentRect(maskW, maskH, origWidth, origHeight)
        val combinedPixels = IntArray(maskW * maskH) { Color.TRANSPARENT }
        val probabilityMasks = mutableListOf<List<List<Float>>>()
        detections.forEach { det ->
            val color = ultralyticsColors[det.cls % ultralyticsColors.size]
            val pm = Array(maskH) { FloatArray(maskW) }
            for (y in 0 until maskH) {
                for (x in 0 until maskW) {
                    var v = 0f
                    for (c in 0 until maskConfidenceLength) {
                        v += det.maskCoeffs[c] * protos[y][x][c]
                    }
                    pm[y][x] = v
                }
            }
            for (y in 0 until maskH) {
                for (x in 0 until maskW) {
                    if (pm[y][x] > threshold) {
                        combinedPixels[y * maskW + x] = color
                    }
                }
            }
            probabilityMasks.add((contentRect.top until contentRect.bottom).map { y ->
                pm[y].copyOfRange(contentRect.left, contentRect.right).toList()
            })
        }
        val bmp = Bitmap.createBitmap(maskW, maskH, Bitmap.Config.ARGB_8888)
        bmp.setPixels(combinedPixels, 0, maskW, 0, 0, maskW, maskH)
        val croppedBmp = if (
            contentRect.left == 0 &&
            contentRect.top == 0 &&
            contentRect.right == maskW &&
            contentRect.bottom == maskH
        ) {
            bmp
        } else {
            Bitmap.createBitmap(
                bmp,
                contentRect.left,
                contentRect.top,
                contentRect.width(),
                contentRect.height()
            ).also { bmp.recycle() }
        }
        return Pair(croppedBmp, probabilityMasks)
    }

    private fun maskContentRect(maskW: Int, maskH: Int, origWidth: Int, origHeight: Int): Rect {
        // Remove prototype-space letterbox padding before masks are scaled to the original image.
        val modelWidth = modelInputSize.first.toFloat()
        val modelHeight = modelInputSize.second.toFloat()
        if (modelWidth <= 0f || modelHeight <= 0f || origWidth <= 0 || origHeight <= 0) {
            return Rect(0, 0, maskW, maskH)
        }

        val gain = min(modelWidth / origWidth, modelHeight / origHeight)
        if (gain <= 0f) return Rect(0, 0, maskW, maskH)
        val resizedWidth = (origWidth * gain).roundToInt()
        val resizedHeight = (origHeight * gain).roundToInt()
        // Match Ultralytics LetterBox leading-pad rounding: round(d - 0.1).
        val padX = ((modelWidth - resizedWidth) / 2f - 0.1f).roundToInt()
        val padY = ((modelHeight - resizedHeight) / 2f - 0.1f).roundToInt()
        val scaleX = maskW / modelWidth
        val scaleY = maskH / modelHeight
        val left = (padX * scaleX).roundToInt().coerceIn(0, maskW - 1)
        val top = (padY * scaleY).roundToInt().coerceIn(0, maskH - 1)
        val right = ((padX + resizedWidth) * scaleX).roundToInt().coerceIn(left + 1, maskW)
        val bottom = ((padY + resizedHeight) * scaleY).roundToInt().coerceIn(top + 1, maskH)
        return Rect(left, top, right, bottom)
    }

    data class Detection(
        val box: RectF,
        val cls: Int,
        val score: Float,
        val maskCoeffs: FloatArray
    )

    /**
     * Load labels from FlatBuffers metadata
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
                            labels = namesMap.values.toList()
                            return true
                        }
                    }
                }
            }
        }
        false
    } catch (e: Exception) {
        Log.e("Segmenter", "Failed to extract metadata: ${e.message}")
        false
    }
}
