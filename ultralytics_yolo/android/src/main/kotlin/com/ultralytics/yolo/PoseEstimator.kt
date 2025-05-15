// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.*
import android.util.Log
import android.util.Size
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
import kotlin.math.max
import kotlin.math.min

class PoseEstimator(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private val confidenceThreshold: Float = 0.25f,   // 任意で変更
    private val iouThreshold: Float = 0.45f           // 任意で変更
) : BasePredictor() {

    companion object {
        // xywh(4) + conf(1) + keypoints(17*3=51) = 56
        private const val OUTPUT_FEATURES = 56
        private const val KEYPOINTS_COUNT = 17
        private const val KEYPOINTS_FEATURES = KEYPOINTS_COUNT * 3 // x, y, conf per keypoint
    }

    private val interpreterOptions = Interpreter.Options().apply {
        if (useGpu) {
            try {
                addDelegate(GpuDelegate())
            } catch (e: Exception) {
                Log.e("PoseEstimator", "GPU delegate error: ${e.message}")
            }
        }
    }

    private lateinit var imageProcessorCamera: ImageProcessor
    private lateinit var imageProcessorSingleImage: ImageProcessor

    init {
        // (1) TFLiteモデルをロード (拡張子自動付与)
        val modelBuffer = YoloUtils.loadModelFile(context, modelPath)

        // ===== メタデータ読み込み（必要に応じて） =====
        try {
            val metadataExtractor = MetadataExtractor(modelBuffer)
            val modelMetadata: ModelMetadata? = metadataExtractor.modelMetadata
            if (modelMetadata != null) {
                Log.d("PoseEstimator", "Model metadata retrieved successfully.")
            }
            val associatedFiles = metadataExtractor.associatedFileNames
            if (!associatedFiles.isNullOrEmpty()) {
                for (fileName in associatedFiles) {
                    Log.d("PoseEstimator", "Found associated file: $fileName")
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
                                    Log.d("PoseEstimator", "Loaded labels from metadata: $labels")
                                } else {

                                }
                            } else {

                            }
                        } catch (ex: Exception) {
                            Log.e("PoseEstimator", "Failed to parse YAML from metadata: ${ex.message}")
                        }
                    }
                }
            } else {
                Log.d("PoseEstimator", "No associated files found in the metadata.")
            }
        } catch (e: Exception) {
            Log.e("PoseEstimator", "Failed to extract metadata: ${e.message}")
        }

        // (2) Interpreterの生成
        interpreter = Interpreter(modelBuffer, interpreterOptions)

        // (3) 入力テンソル形状を取得: [1, height, width, 3]
        val inputShape = interpreter.getInputTensor(0).shape()
        val inHeight = inputShape[1]
        val inWidth = inputShape[2]
        inputSize = com.ultralytics.yolo.Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)

        // (4) ImageProcessorの初期化 - both with and without rotation
        
        // For camera feed (with rotation)
        imageProcessorCamera = ImageProcessor.Builder()
            .add(Rot90Op(3)) // 必要に応じて回転する場合は数値を変えてください
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .add(CastOp(DataType.FLOAT32))
            .build()
            
        // For single images (no rotation needed)
        imageProcessorSingleImage = ImageProcessor.Builder()
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .add(CastOp(DataType.FLOAT32))
            .build()
    }

    override fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean): YOLOResult {
        t0 = System.nanoTime()
        // (1) 前処理: TensorImageへロード & ImageProcessorで処理
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        
        // Choose the appropriate processor based on input source
        val processedImage = if (rotateForCamera) {
            // Apply rotation for camera feed
            imageProcessorCamera.process(tensorImage)
        } else {
            // No rotation for single image
            imageProcessorSingleImage.process(tensorImage)
        }
        val inputBuffer = processedImage.buffer

        // (2) 出力用配列 [1, 56, N] (N=2100等)
        val outputShape = interpreter.getOutputTensor(0).shape()  // 例: [1, 56, 2100]
        val batchSize = outputShape[0]           // 1
        val outFeatures = outputShape[1]         // 56
        val numAnchors = outputShape[2]          // 2100等
        require(outFeatures == OUTPUT_FEATURES) {
            "Unexpected output feature size. Expected=$OUTPUT_FEATURES, Actual=$outFeatures"
        }

        // 出力バッファ作成
        val outputArray = Array(batchSize) {
            Array(outFeatures) { FloatArray(numAnchors) }
        }

        // (3) 推論実行
        interpreter.run(inputBuffer, outputArray)
        // 処理時間の計測更新
        updateTiming()

        // (4) 後処理: NMS + キーポイント計算
        val rawDetections = postProcessPose(
            features = outputArray[0],  // shape: [56][numAnchors]
            numAnchors = numAnchors,
            confidenceThreshold = confidenceThreshold,
            iouThreshold = iouThreshold,
            origWidth = origWidth,
            origHeight = origHeight
        )

        // 検出結果を取り出し
        val boxes = rawDetections.map { it.box }
        val keypointsList = rawDetections.map { it.keypoints }

        // アノテーション用ビットマップ生成
//        val annotatedImage = drawPoseOnBitmap(bitmap, keypointsList, boxes)

        val fpsDouble: Double = if (t4 > 0) (1.0 / t4) else 0.0
        // YOLOResultに詰めて返す
        return YOLOResult(
            origShape = com.ultralytics.yolo.Size(bitmap.height, bitmap.width),
            boxes = boxes,
            keypointsList = keypointsList,
//            annotatedImage = annotatedImage,
            speed = t2,   // ミリ秒等の測定値はBasePredictor側の実装次第
            fps = fpsDouble,
            names = labels
        )
    }

    /**
     * 後処理: Confidence閾値で除外、NMSで抑制、座標変換、Keypoints処理
     */
    private fun postProcessPose(
        features: Array<FloatArray>,
        numAnchors: Int,
        confidenceThreshold: Float,
        iouThreshold: Float,
        origWidth: Int,
        origHeight: Int
    ): List<PoseDetection> {

        val detections = mutableListOf<PoseDetection>()

        // 例えば modelInputSize = (640, 640) と仮定
        val (modelW, modelH) = modelInputSize
        val scaleX = origWidth.toFloat() / modelW
        val scaleY = origHeight.toFloat() / modelH

        for (j in 0 until numAnchors) {
            // 例: features[0][j] ~ features[3][j] は 0～1 の正規化値
            val rawX = features[0][j]       // 0..1
            val rawY = features[1][j]       // 0..1
            val rawW = features[2][j]       // 0..1
            val rawH = features[3][j]       // 0..1
            val conf = features[4][j]       // 0..1

            if (conf < confidenceThreshold) continue

            // (A) まず正規化をモデル入力解像度に拡大 (640等) する
            val xScaled = rawX * modelW
            val yScaled = rawY * modelH
            val wScaled = rawW * modelW
            val hScaled = rawH * modelH

            // [x-w/2, y-h/2, x+w/2, y+h/2] は modelInputSize スケール
            val left   = xScaled - wScaled / 2f
            val top    = yScaled - hScaled / 2f
            val right  = xScaled + wScaled / 2f
            val bottom = yScaled + hScaled / 2f

            // modelInputSize スケールでの RectF
            val normBox = RectF(left / modelW, top / modelH, right / modelW, bottom / modelH)

            // (B) 今度は実際の元画像スケールへ拡大
            val rectF = RectF(
                left   * scaleX,
                top    * scaleY,
                right  * scaleX,
                bottom * scaleY
            )

            // キーポイント (5..55) も同様にモデル解像度 → 実画像スケールに変換
            val kpArray = mutableListOf<Pair<Float, Float>>()
            val kpConfArray = mutableListOf<Float>()
            for (k in 0 until KEYPOINTS_COUNT) {
                val rawKx = features[5 + k * 3][j] // 0..1
                val rawKy = features[5 + k * 3 + 1][j]
                val kpC   = features[5 + k * 3 + 2][j]

                // モデル入力解像度スケールへ
                val kxScaled = rawKx * modelW
                val kyScaled = rawKy * modelH

                // 元画像スケールへ
                val finalKx = kxScaled * scaleX
                val finalKy = kyScaled * scaleY

                kpArray.add(finalKx to finalKy)
                kpConfArray.add(kpC)
            }

            val keypointsObj = Keypoints(
                xyn = kpArray.map { (fx, fy) ->
                    // 0～1 に再正規化する場合: (fx / origWidth, fy / origHeight)
                    (fx / origWidth) to (fy / origHeight)
                },
                xy = kpArray,      // 実画像座標
                conf = kpConfArray
            )

            detections.add(
                PoseDetection(
                    box = Box(index = 0, cls = "person", conf = conf, xywh = rectF, xywhn = normBox),
                    keypoints = keypointsObj
                )
            )
        }

        // 以降は NMS 処理 (単一クラス想定) は変わらず
        val finalDetections = nmsPoseDetections(detections, iouThreshold)
        return finalDetections
    }


    /**
     * NMSを単一クラス想定で実行 (拡張したい場合はクラス別に分割してNMS)
     */
    private fun nmsPoseDetections(
        detections: List<PoseDetection>,
        iouThreshold: Float
    ): List<PoseDetection> {
        // 信頼度で降順ソート
        val sorted = detections.sortedByDescending { it.box.conf }
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

    /**
     * IoU計算
     */
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

    /**
     * キーポイントを描画したビットマップを返す
     */
    private fun drawPoseOnBitmap(
        bitmap: Bitmap,
        keypointsList: List<Keypoints>,
        boxes: List<Box>
    ): Bitmap {
        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)
        val paint = Paint().apply {
            style = Paint.Style.FILL
            color = Color.GREEN
            strokeWidth = 5f
        }
        // バウンディングボックス描画用
        val boxPaint = Paint().apply {
            style = Paint.Style.STROKE
            color = Color.RED
            strokeWidth = 3f
        }

        // 各Personについて描画
        for ((index, person) in keypointsList.withIndex()) {
            // バウンディングボックスの描画
            val boxRect = boxes[index].xywh
            canvas.drawRect(boxRect, boxPaint)

            // キーポイントの描画
            for ((i, kp) in person.xy.withIndex()) {
                // confが一定以上のみ可視化したければ適宜判定
                if (person.conf[i] > 0.25f) {
                    canvas.drawCircle(kp.first, kp.second, 8f, paint)
                }
            }
            // 必要に応じてスケルトン(骨格線)描画を追加
        }
        return output
    }

    /**
     * PoseDetection データクラス
     */
    private data class PoseDetection(
        val box: Box,
        val keypoints: Keypoints
    )
}
