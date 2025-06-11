// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.*
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.tensorflow.lite.support.metadata.schema.ModelMetadata
import org.yaml.snakeyaml.Yaml
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

class ObbDetector(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private val customOptions: Interpreter.Options? = null
) : BasePredictor() {

    private val interpreterOptions: Interpreter.Options = (customOptions ?: Interpreter.Options()).apply {
        // If no custom options provided, use default threads
        if (customOptions == null) {
            setNumThreads(Runtime.getRuntime().availableProcessors())
        }
        
        if (useGpu) {
            try {
                addDelegate(GpuDelegate())
                Log.d("ObbDetector", "GPU delegate is used.")
            } catch (e: Exception) {
                Log.e("ObbDetector", "GPU delegate error: ${e.message}")
            }
        }
    }

    // Similar to PoseEstimator, use ImageProcessor - separate ones for camera portrait/landscape and single images
    private lateinit var imageProcessorCameraPortrait: ImageProcessor
    private lateinit var imageProcessorCameraLandscape: ImageProcessor
    private lateinit var imageProcessorSingleImage: ImageProcessor
    
    // Reuse ByteBuffer for input to reduce allocations
    private lateinit var inputBuffer: ByteBuffer
    
    // Reuse output arrays to reduce allocations
    private lateinit var rawOutput: Array<Array<FloatArray>>
    
    // Transposed array to avoid recreating on each inference
    private lateinit var transposedOutput: Array<FloatArray>
    
    // Output dimensions
    private var outBatch = 0
    private var outChannels = 0
    private var outAnchors = 0

    init {
        val modelBuffer = YOLOUtils.loadModelFile(context, modelPath)

        // ===== Load label information (try Appended ZIP → FlatBuffers in order) =====
        var loadedLabels = YOLOFileUtils.loadLabelsFromAppendedZip(context, modelPath)
        var labelsWereLoaded = loadedLabels != null

        if (labelsWereLoaded) {
            this.labels = loadedLabels!! // Use labels from appended ZIP
            Log.i("ObbDetector", "Labels successfully loaded from appended ZIP.")
        } else {
            Log.w("ObbDetector", "Could not load labels from appended ZIP, trying FlatBuffers metadata...")
            // Try FlatBuffers as a fallback
            if (loadLabelsFromFlatbuffers(modelBuffer)) {
                labelsWereLoaded = true
                Log.i("ObbDetector", "Labels successfully loaded from FlatBuffers metadata.")
            }
        }

        if (!labelsWereLoaded) {
            Log.w("ObbDetector", "No embedded labels found from appended ZIP or FlatBuffers. Using labels passed via constructor (if any) or an empty list.")
            if (this.labels.isEmpty()) {
                Log.w("ObbDetector", "Warning: No labels loaded and no labels provided via constructor. Detections might lack class names.")
            }
        }

        interpreter = Interpreter(modelBuffer, interpreterOptions)
        // Call allocateTensors() once during initialization, not in the inference loop
        interpreter.allocateTensors()
        Log.d("ObbDetector", "TFLite model loaded and tensors allocated")

        val inputShape = interpreter.getInputTensor(0).shape()
        val inHeight = inputShape[1]
        val inWidth = inputShape[2]
        inputSize = Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)
        
        val outShape = interpreter.getOutputTensor(0).shape() // e.g.: [1, outChannels, outAnchors]
        outBatch = outShape[0]       // Usually 1
        outChannels = outShape[1]    // (4 + numClasses + 1)
        outAnchors = outShape[2]     // Total number of anchors
        
        rawOutput = Array(outBatch) {
            Array(outChannels) { FloatArray(outAnchors) }
        }
        
        transposedOutput = Array(outAnchors) {
            FloatArray(outChannels)
        }
        
        val inputBytes = 1 * inHeight * inWidth * 3 * 4 // FLOAT32 is 4 bytes
        inputBuffer = ByteBuffer.allocateDirect(inputBytes).apply {
            order(ByteOrder.nativeOrder())
        }

        
        // For camera feed in portrait mode (with rotation)
        imageProcessorCameraPortrait = ImageProcessor.Builder()
            .add(Rot90Op(3))  // 270-degree rotation
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))  // Normalize to 0~1
            .add(CastOp(DataType.FLOAT32))
            .build()
            
        // For camera feed in landscape mode (no rotation)
        imageProcessorCameraLandscape = ImageProcessor.Builder()
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))  // Normalize to 0~1
            .add(CastOp(DataType.FLOAT32))
            .build()
            
        // For single images (no rotation)
        imageProcessorSingleImage = ImageProcessor.Builder()
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))  // Normalize to 0~1
            .add(CastOp(DataType.FLOAT32))
            .build()
    }

    override fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean, isLandscape: Boolean): YOLOResult {
        t0 = System.nanoTime()

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        
        // Choose appropriate processor based on input source and orientation
        val processedImage = if (rotateForCamera) {
            // Apply appropriate rotation based on device orientation
            if (isLandscape) {
                imageProcessorCameraLandscape.process(tensorImage)
            } else {
                imageProcessorCameraPortrait.process(tensorImage)
            }
        } else {
            // No rotation for single image
            imageProcessorSingleImage.process(tensorImage)
        }
        
        inputBuffer.clear()
        inputBuffer.put(processedImage.buffer)
        inputBuffer.rewind()

        interpreter.run(inputBuffer, rawOutput)
        updateTiming()

        for (i in 0 until outAnchors) {
            for (c in 0 until outChannels) {
                transposedOutput[i][c] = rawOutput[0][c][i]
            }
        }

        val obbDetections = postProcessOBB(
            detections2D = transposedOutput,
            confidenceThreshold = CONFIDENCE_THRESHOLD,
            iouThreshold = IOU_THRESHOLD
        )

        val annotatedImage = drawOBBsOnBitmap(bitmap, obbDetections)

        return YOLOResult(
            origShape = Size(origWidth, origHeight),
            obb = obbDetections,
            annotatedImage = annotatedImage,
            speed = t2,
            fps = if (t4 > 0) 1.0 / t4 else 0.0,
            names = labels
        )
    }


    private fun postProcessOBB(
        detections2D: Array<FloatArray>,
        confidenceThreshold: Float,
        iouThreshold: Float
    ): List<OBBResult> {
        val anchorsCount = detections2D.size
        val numClasses = labels.size

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
            val angle = data[angleIndex]

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
                box = d.obb,
                confidence = d.score,
                cls = labels.getOrElse(d.cls) { "Unknown" },
                index = d.cls
            )
        }
    }

    data class Detection(val obb: OBB, val score: Float, val cls: Int)

    private fun drawOBBsOnBitmap(bitmap: Bitmap, obbDetections: List<OBBResult>): Bitmap {
        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        val paint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 3f
        }
        for (detection in obbDetections) {
            paint.color = ultralyticsColors[detection.index % ultralyticsColors.size]

            val poly = detection.box.toPolygon().map {
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

                paint.style = Paint.Style.FILL
                paint.textSize = 40f
                canvas.drawText(
                    "${detection.cls} ${"%.2f".format(detection.confidence * 100)}%",
                    poly[0].x,
                    poly[0].y - 10,
                    paint
                )
                paint.style = Paint.Style.STROKE
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
    
    /**
     * Load labels from FlatBuffers metadata
     */
    private fun loadLabelsFromFlatbuffers(buf: MappedByteBuffer): Boolean = try {
        val extractor = MetadataExtractor(buf)
        val files = extractor.associatedFileNames
        if (!files.isNullOrEmpty()) {
            for (fileName in files) {
                Log.d("ObbDetector", "Found associated file: $fileName")
                extractor.getAssociatedFile(fileName)?.use { stream ->
                    val fileString = String(stream.readBytes(), Charsets.UTF_8)
                    Log.d("ObbDetector", "Associated file contents:\n$fileString")

                    val yaml = Yaml()
                    @Suppress("UNCHECKED_CAST")
                    val data = yaml.load<Map<String, Any>>(fileString)
                    if (data != null && data.containsKey("names")) {
                        val namesMap = data["names"] as? Map<Int, String>
                        if (namesMap != null) {
                            labels = namesMap.values.toList()
                            Log.d("ObbDetector", "Loaded labels from metadata: $labels")
                            return true
                        }
                    }
                }
            }
        } else {
            Log.d("ObbDetector", "No associated files found in the metadata.")
        }
        false
    } catch (e: Exception) {
        Log.e("ObbDetector", "Failed to extract metadata: ${e.message}")
        false
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
