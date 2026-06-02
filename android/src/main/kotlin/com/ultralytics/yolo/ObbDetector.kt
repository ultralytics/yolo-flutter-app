// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.*
import android.util.Log
import kotlin.math.abs
import kotlin.math.max

class ObbDetector(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private var numItemsThreshold: Int = 30
) : BasePredictor() {

    // Reusable float input for the CompiledModel input buffer.
    private lateinit var floatInput: FloatArray
    private lateinit var inputBitmap: Bitmap
    private lateinit var intValues: IntArray
    
    // Reuse output arrays to reduce allocations
    private lateinit var rawOutput: Array<Array<FloatArray>>
    
    // Transposed array to avoid recreating on each inference
    private lateinit var transposedOutput: Array<FloatArray>
    
    // Output dimensions
    private var outBatch = 0
    private var outChannels = 0
    private var outAnchors = 0

    init {
        val loadedLabels = YOLOFileUtils.loadModelLabels(context, modelPath)
        if (loadedLabels != null) {
            this.labels = loadedLabels
            Log.i("ObbDetector", "Labels successfully loaded from appended ZIP.")
        } else if (this.labels.isEmpty()) {
            Log.w("ObbDetector", "No embedded labels found and none provided; detections may lack class names.")
        }

        rtModel = LiteRtModel(modelPath, useGpu, "ObbDetector")

        val inDims = rtModel.inputDims
        val inHeight = if (inDims.size >= 4) inDims[1] else 640
        val inWidth = if (inDims.size >= 4) inDims[2] else 640
        inputSize = Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)

        val outShape = rtModel.outputDims.getOrNull(0) ?: IntArray(0) // [1, outChannels, outAnchors]
        outBatch = if (outShape.isNotEmpty()) outShape[0] else 1
        outChannels = if (outShape.size >= 3) outShape[1] else 0
        outAnchors = if (outShape.size >= 3) outShape[2] else 0

        rawOutput = Array(outBatch) {
            Array(outChannels) { FloatArray(outAnchors) }
        }

        transposedOutput = Array(outAnchors) {
            FloatArray(outChannels)
        }

        floatInput = FloatArray(inWidth * inHeight * 3)
        inputBitmap = Bitmap.createBitmap(inWidth, inHeight, Bitmap.Config.ARGB_8888)
        intValues = IntArray(inWidth * inHeight)
    }
    
    override fun setNumItemsThreshold(n: Int) {
        this.numItemsThreshold = n
        super.setNumItemsThreshold(n)
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
        ImageUtils.copyRgbBitmapToFloatArray(inputBitmap, floatInput, intValues)

        // Reshape the flat output into rawOutput[0][outChannels][outAnchors].
        val flat = rtModel.run(floatInput)[0]
        var idx = 0
        for (c in 0 until outChannels) {
            val row = rawOutput[0][c]
            for (a in 0 until outAnchors) {
                row[a] = flat[idx++]
            }
        }
        for (i in 0 until outAnchors) {
            for (c in 0 until outChannels) {
                transposedOutput[i][c] = rawOutput[0][c][i]
            }
        }

        val obbDetections = postProcessOBB(
            detections2D = transposedOutput,
            confidenceThreshold = CONFIDENCE_THRESHOLD,
            iouThreshold = IOU_THRESHOLD,
            origWidth = origWidth,
            origHeight = origHeight
        )
        
        // Apply numItemsThreshold limit
        val limitedDetections = obbDetections.take(numItemsThreshold)

        val annotatedImage = if (rotateForCamera) null else {
            drawOBBsOnBitmap(bitmap, limitedDetections, origWidth, origHeight)
        }

        val timing = finishTiming()
        return YOLOResult(
            origShape = Size(origWidth, origHeight),
            obb = limitedDetections,
            annotatedImage = annotatedImage,
            speed = timing.speedMs,
            fps = timing.fps,
            names = labels
        )
    }


    private fun postProcessOBB(
        detections2D: Array<FloatArray>,
        confidenceThreshold: Float,
        iouThreshold: Float,
        origWidth: Int,
        origHeight: Int
    ): List<OBBResult> {
        val anchorsCount = detections2D.size
        val numClasses = ((detections2D.firstOrNull()?.size ?: 5) - 5).coerceAtLeast(0)

        val detections = mutableListOf<Detection>()

        for (i in 0 until anchorsCount) {
            val data = detections2D[i]
            val cx = data[0]
            val cy = data[1]
            val w  = data[2]
            val h  = data[3]

            // Check class scores
            var bestScore = 0f
            var bestClass = 0
            for (c in 0 until numClasses) {
                val score = data[4 + c]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }

            val angleIndex = 4 + numClasses
            val angle = if (angleIndex < data.size) data[angleIndex] else 0f

            if (bestScore >= confidenceThreshold) {
                val obb = OBB(cx, cy, w, h, angle)
                detections.add(Detection(obb, bestScore, bestClass))
            }
        }

        // === NMS ===
        val boxes = detections.map { it.obb }
        val scores = detections.map { it.score }
        val keepIndices = nonMaxSuppressionOBB(boxes, scores, iouThreshold)

        return keepIndices.map { idx ->
            val d = detections[idx]
            OBBResult(
                box = inputOBBFromModelOBB(d.obb, origWidth, origHeight),
                confidence = d.score,
                cls = labelName(d.cls),
                index = d.cls
            )
        }
    }

    data class Detection(val obb: OBB, val score: Float, val cls: Int)

    private fun drawOBBsOnBitmap(
        bitmap: Bitmap,
        obbDetections: List<OBBResult>,
        origWidth: Int,
        origHeight: Int
    ): Bitmap {
        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        val paint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 3f
        }
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = 40f
        }
        val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG)
        for (detection in obbDetections) {
            paint.color = ultralyticsColors[detection.index % ultralyticsColors.size]

            val poly = detection.box.toPolygon(origWidth.toFloat(), origHeight.toFloat()).map {
                PointF(it.x * bitmap.width, it.y * bitmap.height)
            }
            if (poly.size >= 4) {
                val path = Path().apply {
                    moveTo(poly[0].x, poly[0].y)
                    for (p in poly.drop(1)) {
                        lineTo(p.x, p.y)
                    }
                    close()
                }
                canvas.drawPath(path, paint)

                val label = "${detection.cls} ${"%.1f".format(detection.confidence * 100)}"
                val labelWidth = textPaint.measureText(label) + 12f
                val labelHeight = textPaint.textSize
                val labelRect = RectF(
                    poly[0].x - 2f,
                    poly[0].y - labelHeight - 2f,
                    poly[0].x - 2f + labelWidth,
                    poly[0].y - 2f
                )
                labelPaint.color = paint.color
                labelPaint.alpha = 153
                canvas.drawRoundRect(labelRect, 3f, 3f, labelPaint)
                canvas.drawText(label, labelRect.left + 6f, labelRect.bottom - 8f, textPaint)
            }
        }
        return output
    }


    private fun nonMaxSuppressionOBB(
        boxes: List<OBB>,
        scores: List<Float>,
        iouThreshold: Float
    ): List<Int> {
        val sortedIndices = scores.indices.sortedByDescending { scores[it] }
        val keep = mutableListOf<Int>()
        val active = BooleanArray(boxes.size) { true }
        val infoList = boxes.map { OBBInfo(it) }

        for (i in sortedIndices.indices) {
            val idx = sortedIndices[i]
            if (!active[idx]) continue
            keep.add(idx)
            val boxA = infoList[idx]
            for (j in (i + 1) until sortedIndices.size) {
                val idxB = sortedIndices[j]
                if (!active[idxB]) continue
                val boxB = infoList[idxB]
                if (boxA.aabbIntersect(boxB)) {
                    val iouVal = boxA.iou(boxB)
                    if (iouVal > iouThreshold) {
                        active[idxB] = false
                    }
                }
            }
        }
        return keep
    }

    private data class OBBInfo(
        val obb: OBB,
        val polygon: List<PointF>,
        val area: Float,
        val aabb: RectF
    ) {
        constructor(obb: OBB) : this(
            obb,
            obb.toPolygon(),
            obb.area,
            obb.toAABB()
        )

        fun iou(other: OBBInfo): Float {
            val interPoly = polygonIntersection(polygon, other.polygon)
            val interArea = polygonArea(interPoly)
            val unionArea = area + other.area - interArea
            if (unionArea <= 0f) return 0f
            return interArea / unionArea
        }

        fun aabbIntersect(other: OBBInfo): Boolean {
            return RectF.intersects(this.aabb, other.aabb)
        }
    }

    private fun polygonIntersection(subject: List<PointF>, clip: List<PointF>): List<PointF> {
        var outputList = subject
        if (outputList.isEmpty()) return emptyList()
        val clipClosed = if (clip.isNotEmpty() && clip.first() == clip.last()) {
            clip
        } else {
            clip + clip.first()
        }

        for (i in 0 until (clipClosed.size - 1)) {
            val p1 = clipClosed[i]
            val p2 = clipClosed[i + 1]
            val inputList = outputList
            outputList = mutableListOf()
            if (inputList.isEmpty()) break

            val inputClosed = if (inputList.isNotEmpty() && inputList.first() == inputList.last()) {
                inputList
            } else {
                inputList + inputList.first()
            }

            for (j in 0 until (inputClosed.size - 1)) {
                val current = inputClosed[j]
                val next = inputClosed[j + 1]
                val currentInside = isInside(current, p1, p2)
                val nextInside    = isInside(next,    p1, p2)

                if (currentInside && nextInside) {
                    outputList.add(next)
                } else if (currentInside && !nextInside) {
                    val inter = computeIntersection(current, next, p1, p2)
                    if (inter != null) outputList.add(inter)
                } else if (!currentInside && nextInside) {
                    val inter = computeIntersection(current, next, p1, p2)
                    if (inter != null) outputList.add(inter)
                    outputList.add(next)
                }
            }
        }
        return outputList
    }

    private fun isInside(point: PointF, p1: PointF, p2: PointF): Boolean {
        val cross = (p2.x - p1.x) * (point.y - p1.y) -
                (p2.y - p1.y) * (point.x - p1.x)
        return cross >= 0f
    }

    private fun computeIntersection(
        p1: PointF,
        p2: PointF,
        clipStart: PointF,
        clipEnd: PointF
    ): PointF? {
        val x1 = p1.x
        val y1 = p1.y
        val x2 = p2.x
        val y2 = p2.y
        val x3 = clipStart.x
        val y3 = clipStart.y
        val x4 = clipEnd.x
        val y4 = clipEnd.y

        val denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if (kotlin.math.abs(denom) < 1e-7) {
            return null
        }
        val t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        val ix = x1 + t * (x2 - x1)
        val iy = y1 + t * (y2 - y1)
        return PointF(ix.toFloat(), iy.toFloat())
    }

    private fun polygonArea(poly: List<PointF>): Float {
        if (poly.size < 3) return 0f
        var area = 0f
        for (i in 0 until (poly.size - 1)) {
            area += (poly[i].x * poly[i + 1].y) - (poly[i + 1].x * poly[i].y)
        }
        area += (poly.last().x * poly.first().y) - (poly.first().x * poly.last().y)
        return kotlin.math.abs(area) * 0.5f
    }
    
}

/** Extension to get Axis-Aligned Bounding Box (AABB) of OBB */
fun OBB.toAABB(): RectF {
    val poly = toPolygon()
    var minX = Float.POSITIVE_INFINITY
    var maxX = Float.NEGATIVE_INFINITY
    var minY = Float.POSITIVE_INFINITY
    var maxY = Float.NEGATIVE_INFINITY
    for (pt in poly) {
        if (pt.x < minX) minX = pt.x
        if (pt.x > maxX) maxX = pt.x
        if (pt.y < minY) minY = pt.y
        if (pt.y > maxY) maxY = pt.y
    }
    return RectF(minX, minY, maxX, maxY)
}
