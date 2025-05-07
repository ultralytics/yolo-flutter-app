package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
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

class Classifier(
    context: Context,
    modelPath: String,
    // fallback 用にコンストラクタ引数からもラベルを受け取れるようにしておく
    override var labels: List<String> = emptyList(),
    private val useGpu: Boolean = true
) : BasePredictor() {

    private val interpreterOptions: Interpreter.Options = Interpreter.Options().apply {
        if (useGpu) {
            try {
                addDelegate(GpuDelegate())
                Log.d(TAG, "GPU delegate is used.")
            } catch (e: Exception) {
                Log.e(TAG, "GPU delegate error: ${e.message}")
            }
        }
        // 必要ならスレッド数指定など
        setNumThreads(4)
    }

    var numClass: Int = 0

    // 画像の前処理をまとめて行うためのパイプライン
    private lateinit var imageProcessorCamera: ImageProcessor
    private lateinit var imageProcessorSingleImage: ImageProcessor

    init {
        val modelBuffer = YoloUtils.loadModelFile(context, modelPath)

        // ===== メタデータから labels を読み込み (存在すれば) =====
        try {
            val metadataExtractor = MetadataExtractor(modelBuffer)
            val modelMetadata: ModelMetadata? = metadataExtractor.modelMetadata
            if (modelMetadata != null) {
                Log.d(TAG, "Model metadata retrieved successfully.")
            }

            // メタデータに関連付けられたファイル一覧を取得
            val associatedFiles = metadataExtractor.associatedFileNames
            if (!associatedFiles.isNullOrEmpty()) {
                for (fileName in associatedFiles) {
                    Log.d(TAG, "Found associated file: $fileName")
                    val inputStream = metadataExtractor.getAssociatedFile(fileName)
                    inputStream?.use { stream ->
                        val fileContent = stream.readBytes()
                        val fileString = fileContent.toString(Charsets.UTF_8)
                        Log.d(TAG, "Associated file contents:\n$fileString")

                        // YAML をパースして "names" があればラベルとして取得
                        try {
                            val yaml = Yaml()
                            @Suppress("UNCHECKED_CAST")
                            val data = yaml.load<Map<String, Any>>(fileString)
                            if (data != null && data.containsKey("names")) {
                                val namesMap = data["names"] as? Map<Int, String>
                                if (namesMap != null) {
                                    this.labels = namesMap.values.toList()
                                    Log.d(TAG, "Loaded labels from metadata: $labels")
                                } else {}
                            } else {}
                        } catch (ex: Exception) {
                            Log.e(TAG, "Failed to parse YAML from metadata: ${ex.message}")
                        }
                    }
                }
            } else {
                Log.d(TAG, "No associated files found in the metadata.")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to extract metadata: ${e.message}")
        }

        // Interpreter の生成
        interpreter = Interpreter(modelBuffer, interpreterOptions)

        // 入力テンソル形状 [1, height, width, 3] を取得
        val inputShape = interpreter.getInputTensor(0).shape()
        // 例えば inputShape = [1, 224, 224, 3]
        val inBatch = inputShape[0]   // 通常 1
        val inHeight = inputShape[1]
        val inWidth = inputShape[2]
        val inChannels = inputShape[3]
        require(inBatch == 1 && inChannels == 3) {
            "Unexpected input tensor shape. Expect [1,H,W,3], but got ${inputShape.joinToString()}"
        }
        // BasePredictor 側の変数に記憶
        inputSize = Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)
        Log.d(TAG, "Model input size = $inWidth x $inHeight")

        // 出力テンソル形状 [1, numClass] を想定
        val outputShape = interpreter.getOutputTensor(0).shape()
        // 例えば outputShape = [1, 1000]
        numClass = outputShape[1]
        Log.d(TAG, "Model output shape = [1, $numClass]")

        // ===== 画像前処理パイプラインの用意 =====
        // For camera feed (with rotation)
        imageProcessorCamera = ImageProcessor.Builder()
            .add(Rot90Op(3))  // 必要に応じて回転
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(INPUT_MEAN, INPUT_STD))  // [0,1]にスケーリング
            .add(CastOp(DataType.FLOAT32))
            .build()
            
        // For single images (no rotation)
        imageProcessorSingleImage = ImageProcessor.Builder()
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(INPUT_MEAN, INPUT_STD))  // [0,1]にスケーリング
            .add(CastOp(DataType.FLOAT32))
            .build()

        Log.d(TAG, "Classifier initialized.")
    }

    override fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean): YOLOResult {
        // 計測開始
        t0 = System.nanoTime()

        // ======== 前処理 ========
        // TFLite Support Library を使う流れ
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)              // Bitmap を読み込み
        
        // Choose appropriate processor based on input source
        val processedImage = if (rotateForCamera) {
            // Apply rotation for camera feed
            imageProcessorCamera.process(tensorImage)
        } else {
            // No rotation for single image
            imageProcessorSingleImage.process(tensorImage)
        }
        val inputBuffer = processedImage.buffer

        // ======== 推論 ========
        // 出力 shape = [1, numClass]
        val outputArray = Array(1) { FloatArray(numClass) }
        interpreter.run(inputBuffer, outputArray)

        // 計測終了
        updateTiming()   // 内部で t2(ms), t4(sec) などが更新される想定

        // ======== 後処理: スコア順に並べて top1, top5 を取得 ========
        val scores = outputArray[0]   // FloatArray(numClass)
        val indexedScores = scores.mapIndexed { index, score -> index to score }
        val sorted = indexedScores.sortedByDescending { it.second }

        // Top1
        val top1 = sorted.firstOrNull()
        // Top5
        val top5 = sorted.take(5)

        val top1Label = if (top1 != null) labels.getOrElse(top1.first) { "Unknown" } else "Unknown"
        val top1Score = top1?.second ?: 0f
        val top1Index: Int = if (top1 != null) top1.first else 0

        val top5Labels = top5.map { (idx, _) -> labels.getOrElse(idx) { "Unknown" } }
        val top5Scores = top5.map { it.second }

        // YOLOResult の Probs に詰める
        val probs = Probs(
            top1 = top1Label,
            top5 = top5Labels,
            top1Conf = top1Score,
            top5Confs = top5Scores,
            top1Index = top1Index
        )

        // fps は if(t4>0) 1.0/t4 else 0.0 など。BasePredictor 側の実装に合わせて
        val fpsVal = if (t4 > 0) 1.0 / t4 else 0.0

        return YOLOResult(
            origShape = Size(bitmap.width, bitmap.height), // 元画像サイズ
            probs = probs,
            speed = t2,               // ミリ秒
            fps = fpsVal,
            names = labels
        )
    }

    companion object {
        private const val TAG = "Classifier"

        // ObjectDetector 同様の前処理用定数
        private const val INPUT_MEAN = 0f
        private const val INPUT_STD = 255f
    }
}
