// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.*
import android.util.Log
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class Segmenter(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private var numItemsThreshold: Int = 30
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

    // Reusable float input for the CompiledModel input buffer.
    private lateinit var floatInput: FloatArray
    private lateinit var inputBitmap: Bitmap
    private lateinit var intValues: IntArray

    // CompiledModel output indices: detection head (rank-3) and mask proto (rank-4) may be returned in either order.
    private var detOutIndex = 0
    private var maskOutIndex = 1

    init {
        val loadedLabels = YOLOFileUtils.loadModelLabels(context, modelPath)
        if (loadedLabels != null) {
            this.labels = loadedLabels
            Log.i("Segmenter", "Labels successfully loaded from appended ZIP.")
        } else if (this.labels.isEmpty()) {
            Log.w("Segmenter", "No embedded labels found and none provided; detections may lack class names.")
        }

        rtModel = LiteRtModel(modelPath, useGpu, "Segmenter")

        val inDims = rtModel.inputDims
        val inHeight = if (inDims.size >= 4) inDims[1] else 640
        val inWidth = if (inDims.size >= 4) inDims[2] else 640
        inputSize = com.ultralytics.yolo.Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)

        // Segment has two outputs returned in arbitrary order: the detection head (rank 3) and the mask proto (rank 4).
        val dims = rtModel.outputDims
        detOutIndex = dims.indexOfFirst { it.size == 3 }.takeIf { it >= 0 } ?: 0
        maskOutIndex = dims.indexOfFirst { it.size == 4 }.takeIf { it >= 0 } ?: 1
        val out0Shape = dims.getOrNull(detOutIndex) ?: IntArray(0)
        val out1Shape = dims.getOrNull(maskOutIndex) ?: IntArray(0)

        // Detection head shape (traditional [1,116,8400] or end-to-end [1,300,38]); kept flat and indexed in place.
        isEndToEnd = out0Shape[2] < out0Shape[1] && out0Shape[2] >= 6
        if (isEndToEnd) {
            out0NumAnchors = out0Shape[1]
            out0NumFeatures = out0Shape[2]
        } else {
            out0NumFeatures = out0Shape[1]
            out0NumAnchors = out0Shape[2]
        }

        // Mask proto shape (example: [1,160,160,32]); kept flat and indexed in place.
        maskH = out1Shape[1]
        maskW = out1Shape[2]
        maskC = out1Shape[3]

        floatInput = FloatArray(inWidth * inHeight * 3)
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
        ImageUtils.copyRgbBitmapToFloatArray(inputBitmap, floatInput, intValues)

        numClasses = out0NumFeatures - boxFeatureLength - maskConfidenceLength

        val outs = try {
            rtModel.run(floatInput)
        } catch (e: Exception) {
            Log.e("Segmenter", "Inference error: ${e.message}")
            val timing = finishTiming()
            return YOLOResult(
                origShape = com.ultralytics.yolo.Size(origWidth, origHeight),
                boxes = emptyList(),
                speed = timing.speedMs,
                fps = timing.fps,
                names = labels
            )
        }

        // Index the flat run() outputs directly — no per-frame reshape into jagged nested arrays.
        val detFlat = outs[detOutIndex]
        val maskFlat = outs[maskOutIndex]
        // (4) Post-processing (box + mask)
        val rawDetections = postProcessSegment(
            detFlat = detFlat,
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
            boxes.add(Box(cls, labelName(cls), score, rectF, normRect))
            maskDetections.add(detection)
        }

        val (combinedMask, probMasks) = generateCombinedMaskImage(
            detections = maskDetections,
            protoFlat = maskFlat,
            maskW = maskW,
            maskH = maskH,
            origWidth = origWidth,
            origHeight = origHeight,
            returnIndividualMasks = includeRawMaskData
        )
        val masks = Masks(probMasks ?: emptyList(), combinedMask)
        val timing = finishTiming()
        return YOLOResult(
            origShape = com.ultralytics.yolo.Size(origWidth, origHeight),
            boxes = boxes,
            masks = masks,
            speed = timing.speedMs,
            fps = timing.fps,
            names = labels
        )
    }

    private fun postProcessSegment(
        detFlat: FloatArray,
        numAnchors: Int,
        confidenceThreshold: Float,
        iouThreshold: Float
    ): List<Detection> {
        if (isEndToEnd) {
            return postProcessEndToEndSegment(detFlat, confidenceThreshold)
        }

        // The detection head is flat in feature-major order: feature[f][anchor] == detFlat[f * numAnchors + anchor].
        // Estimated capacity for results list to reduce reallocations
        val estimatedCapacity = (numAnchors * 0.05).toInt() // Assume ~5% will pass threshold
        val results = ArrayList<Detection>(estimatedCapacity)

        // Apply early filtering - Optimization: early pruning strategy
        val earlyThreshold = confidenceThreshold * 0.8f // Slightly lower threshold for first pass
        val classBase = 4 * numAnchors

        for (j in 0 until numAnchors) {
            // Check all classes instead of just first 3 to avoid bias
            var quickMaxScore = 0f
            for (c in 0 until numClasses) {
                quickMaxScore = max(quickMaxScore, detFlat[classBase + c * numAnchors + j])
            }

            // Skip further processing if clearly below threshold
            if (quickMaxScore < earlyThreshold) continue

            // Continue with full processing for potential detections
            val cx = detFlat[j]
            val cy = detFlat[numAnchors + j]
            val w = detFlat[2 * numAnchors + j]
            val h = detFlat[3 * numAnchors + j]
            var maxScore = 0f
            var maxClassIdx = 0

            for (c in 0 until numClasses) {
                val score = detFlat[classBase + c * numAnchors + j]
                if (score > maxScore) {
                    maxScore = score
                    maxClassIdx = c
                }
            }

            if (maxScore >= confidenceThreshold) {
                val maskCoeffs = FloatArray(maskConfidenceLength)
                val base = (4 + numClasses) * numAnchors
                for (m in 0 until maskConfidenceLength) {
                    maskCoeffs[m] = detFlat[base + m * numAnchors + j]
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
        detFlat: FloatArray,
        confidenceThreshold: Float
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        // End-to-end head is anchor-major: feature[anchor][field] == detFlat[anchor * out0NumFeatures + field].
        val fieldCount = out0NumFeatures
        val maskStart = if (fieldCount > 5) 6 else 5

        for (j in 0 until out0NumAnchors) {
            val o = j * out0NumFeatures
            val confidence = detFlat[o + 4]
            if (confidence < confidenceThreshold) continue

            val maskCoeffs = FloatArray(maskConfidenceLength)
            for (m in 0 until min(maskConfidenceLength, fieldCount - maskStart)) {
                maskCoeffs[m] = detFlat[o + maskStart + m]
            }
            detections.add(
                Detection(
                    RectF(detFlat[o], detFlat[o + 1], detFlat[o + 2], detFlat[o + 3]),
                    if (fieldCount > 5) detFlat[o + 5].toInt() else 0,
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
        protoFlat: FloatArray,
        maskW: Int,
        maskH: Int,
        origWidth: Int,
        origHeight: Int,
        threshold: Float = 0.0f,
        returnIndividualMasks: Boolean = false
    ): Pair<Bitmap?, List<List<List<Float>>>?> {
        if (detections.isEmpty()) return Pair(null, null)
        if (maskW <= 0 || maskH <= 0 || maskC <= 0) return Pair(null, null)
        val contentRect = modelMaskCropRect(maskW, maskH, origWidth, origHeight)
            ?: Rect(0, 0, maskW, maskH)
        val combinedPixels = IntArray(maskW * maskH) { Color.TRANSPARENT }
        val probabilityMasks = if (returnIndividualMasks) mutableListOf<List<List<Float>>>() else null
        val coeffCount = min(maskConfidenceLength, maskC)
        detections.forEach { det ->
            val color = ultralyticsColors[det.cls % ultralyticsColors.size]
            val instanceMask = if (returnIndividualMasks) {
                Array(contentRect.height()) { FloatArray(contentRect.width()) }
            } else {
                null
            }
            val maskBox = maskBoxForDetection(det.box, maskW, maskH)
            val outputBox =
                if (maskBox.left < maskBox.right && maskBox.top < maskBox.bottom) {
                    Rect(
                        max(maskBox.left, contentRect.left),
                        max(maskBox.top, contentRect.top),
                        min(maskBox.right, contentRect.right),
                        min(maskBox.bottom, contentRect.bottom)
                    )
                } else {
                    null
                }
            if (
                outputBox == null ||
                outputBox.left >= outputBox.right ||
                outputBox.top >= outputBox.bottom
            ) {
                instanceMask?.let { mask ->
                    probabilityMasks?.add(mask.map { it.toList() })
                }
                return@forEach
            }
            // proto is HWC flat: protos[y][x][c] == protoFlat[(y * maskW + x) * maskC + c]. Walking c contiguously
            // keeps each coeff dot-product on one cache line.
            for (y in outputBox.top until outputBox.bottom) {
                val rowBase = y * maskW
                for (x in outputBox.left until outputBox.right) {
                    val base = (rowBase + x) * maskC
                    var v = 0f
                    for (c in 0 until coeffCount) {
                        v += det.maskCoeffs[c] * protoFlat[base + c]
                    }
                    if (v > threshold) {
                        combinedPixels[rowBase + x] = color
                        instanceMask?.get(y - contentRect.top)?.set(x - contentRect.left, v)
                    }
                }
            }
            instanceMask?.let { mask ->
                probabilityMasks?.add(mask.map { it.toList() })
            }
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

    private fun maskBoxForDetection(outputBox: RectF, maskW: Int, maskH: Int): Rect {
        val modelBox = modelRectFromOutputRect(outputBox, outputCoordinatesAreNormalized(outputBox))
        val scaleX = maskW.toFloat() / modelInputSize.first
        val scaleY = maskH.toFloat() / modelInputSize.second
        val left = (modelBox.left * scaleX).roundToInt().coerceIn(0, maskW)
        val top = (modelBox.top * scaleY).roundToInt().coerceIn(0, maskH)
        val right = (modelBox.right * scaleX).roundToInt().coerceIn(0, maskW)
        val bottom = (modelBox.bottom * scaleY).roundToInt().coerceIn(0, maskH)
        return Rect(left, top, right, bottom)
    }

    data class Detection(
        val box: RectF,
        val cls: Int,
        val score: Float,
        val maskCoeffs: FloatArray
    )

}
