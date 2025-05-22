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

    // PoseEstimator と同様に、ImageProcessor を使う - one for camera and one for single images
    private lateinit var imageProcessorCamera: ImageProcessor
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
        // 1) TFLiteモデルをロード (拡張子自動付与)
        val modelBuffer = YoloUtils.loadModelFile(context, modelPath)

        // 2) メタデータ読み込み（必要であれば）
        try {
            val metadataExtractor = MetadataExtractor(modelBuffer)
            val modelMetadata: ModelMetadata? = metadataExtractor.modelMetadata
            if (modelMetadata != null) {
                Log.d("ObbDetector", "Model metadata retrieved successfully.")
            }
            val associatedFiles = metadataExtractor.associatedFileNames
            if (!associatedFiles.isNullOrEmpty()) {
                for (fileName in associatedFiles) {
                    Log.d("ObbDetector", "Found associated file: $fileName")
                    metadataExtractor.getAssociatedFile(fileName)?.use { stream ->
                        val fileContent = stream.readBytes()
                        val fileString = fileContent.toString(Charsets.UTF_8)
                        try {
                            val yaml = Yaml()
                            @Suppress("UNCHECKED_CAST")
                            val data = yaml.load<Map<String, Any>>(fileString)
                            if (data != null && data.containsKey("names")) {
                                val namesMap = data["names"] as? Map<Int, String>
                                if (namesMap != null) {
                                    this.labels = namesMap.values.toList()
                                    Log.d("ObbDetector", "Loaded labels from metadata: $labels")
                                } else {}
                            } else {}
                        } catch (ex: Exception) {
                            Log.e("ObbDetector", "Failed to parse YAML from metadata: ${ex.message}")
                        }
                    }
                }
            } else {
                Log.d("ObbDetector", "No associated files found in the metadata.")
            }
        } catch (e: Exception) {
            Log.e("ObbDetector", "Failed to extract metadata: ${e.message}")
        }

        // 3) Interpreter生成
        interpreter = Interpreter(modelBuffer, interpreterOptions)
        // Call allocateTensors() once during initialization, not in the inference loop
        interpreter.allocateTensors()
        Log.d("ObbDetector", "TFLite model loaded and tensors allocated")

        // 4) 入力テンソル形状を取得 (例: [1, height, width, 3])
        val inputShape = interpreter.getInputTensor(0).shape()
        val inHeight = inputShape[1]
        val inWidth = inputShape[2]
        inputSize = Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)
        
        // 出力テンソル形状を取得して初期化
        val outShape = interpreter.getOutputTensor(0).shape() // 例: [1, outChannels, outAnchors]
        outBatch = outShape[0]       // 通常1
        outChannels = outShape[1]    // (4 + numClasses + 1)
        outAnchors = outShape[2]     // アンカー総数
        
        // 出力バッファを一度だけ初期化
        rawOutput = Array(outBatch) {
            Array(outChannels) { FloatArray(outAnchors) }
        }
        
        // 転置配列も事前に初期化
        transposedOutput = Array(outAnchors) {
            FloatArray(outChannels)
        }
        
        // 入力バッファの初期化 (直接確保)
        val inputBytes = 1 * inHeight * inWidth * 3 * 4 // FLOAT32 は4バイト
        inputBuffer = ByteBuffer.allocateDirect(inputBytes).apply {
            order(ByteOrder.nativeOrder())
        }

        // 5) imageProcessor の初期化 - both with and without rotation
        
        // For camera feed (with rotation)
        imageProcessorCamera = ImageProcessor.Builder()
            .add(Rot90Op(3))  // 必要に応じて回転させる場合は 1~3等を設定
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))  // 0~1 に正規化
            .add(CastOp(DataType.FLOAT32))
            .build()
            
        // For single images (no rotation)
        imageProcessorSingleImage = ImageProcessor.Builder()
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))  // 0~1 に正規化
            .add(CastOp(DataType.FLOAT32))
            .build()
    }

    override fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean): YOLOResult {
        t0 = System.nanoTime()

        // === (1) 前処理: TensorImageへロード & ImageProcessorで処理 ===
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        
        // Choose appropriate processor based on input source
        val processedImage = if (rotateForCamera) {
            // Apply rotation for camera feed
            imageProcessorCamera.process(tensorImage)
        } else {
            // No rotation for single image
            imageProcessorSingleImage.process(tensorImage)
        }
        
        // 再利用可能なバッファへ入力をコピー
        inputBuffer.clear()
        inputBuffer.put(processedImage.buffer)
        inputBuffer.rewind()

        // === (3) 推論実行 (出力バッファは初期化時に確保済み) ===
        interpreter.run(inputBuffer, rawOutput)
        updateTiming()

        // === (4) shape 転置して後処理 ===
        // 事前確保済みの配列を再利用
        for (i in 0 until outAnchors) {
            for (c in 0 until outChannels) {
                transposedOutput[i][c] = rawOutput[0][c][i]
            }
        }

        // ここで (i番目) = [cx, cy, w, h, classScores..., angle]
        val obbDetections = postProcessOBB(
            detections2D = transposedOutput,
            confidenceThreshold = CONFIDENCE_THRESHOLD,
            iouThreshold = IOU_THRESHOLD
        )

        // アノテーション用に描画
        val annotatedImage = drawOBBsOnBitmap(bitmap, obbDetections)

        return YOLOResult(
            origShape = Size(bitmap.height, bitmap.width),
            obb = obbDetections,
            annotatedImage = annotatedImage,
            speed = t2,
            fps = if (t4 > 0) 1.0 / t4 else 0.0,
            names = labels
        )
    }

    /**
     * 後処理: [anchorCount][channels] 配列から OBB を取り出し、NMS
     */
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

            // クラススコアを確認
            var bestScore = 0f
            var bestClass = 0
            for (c in 0 until numClasses) {
                val score = data[4 + c]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }

            // 最後が angle
            val angleIndex = 4 + numClasses
            val angle = data[angleIndex]

            // 閾値チェック
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

    /**
     * アノテーション描画 (OBB ポリゴン + ラベル)
     */
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

    // ===============================================================
    // NMSやポリゴン関連のヘルパーは従来通り
    // ===============================================================

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

/** OBB の Axis-Aligned Bounding Box (AABB) を取得する拡張 */
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
