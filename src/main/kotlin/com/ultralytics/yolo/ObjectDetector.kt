package com.ultralytics.yolo

import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.yaml.snakeyaml.Yaml
import java.nio.ByteBuffer

/**
 * 目标检测器 (Object Detector)
 * 负责执行 TFLite 模型推理，包含预处理和后处理逻辑。
 * 
 * @param modelPath 模型文件绝对路径
 * @param useGpu 是否使用 GPU 加速
 * @param numItemsThreshold 最大检测数量限制
 */
class ObjectDetector(
    modelPath: String,
    private val useGpu: Boolean = true,
    private var numItemsThreshold: Int = 30
) {
    private val TAG = "ObjectDetector"

    // TFLite 解释器
    private lateinit var interpreter: Interpreter
    private var labels: List<String> = emptyList()

    // 图像处理器 (缩放 -> 归一化 -> 类型转换)
    private lateinit var imageProcessor: ImageProcessor

    // 输入输出配置
    private var inputHeight = 0
    private var inputWidth = 0
    private var outputChannel = 0 // 例如 84 (xywh + 80 classes)
    private var outputAnchors = 0 // 例如 8400 (anchors)

    // 输入缓冲
    private lateinit var tensorImage: TensorImage
    
    // 输出缓冲
    // YOLO v8/11/26 输出通常是 [1, 84, 8400] -> [batch, channels, anchors]
    private lateinit var outputBuffer: Array<Array<FloatArray>>

    // 阈值配置
    private var confidenceThreshold = 0.25f
    private var iouThreshold = 0.45f

    init {
        // 1. 加载模型 (仅支持绝对路径)
        val modelBuffer = YOLOUtils.loadModelFile(modelPath)
        
        // 2. 初始化解释器选项
        val options = Interpreter.Options().apply {
            setNumThreads(Runtime.getRuntime().availableProcessors())
            setAllowFp16PrecisionForFp32(true)
            if (useGpu) {
                try {
                    addDelegate(GpuDelegate())
                    Log.d(TAG, "已启用 GPU 代理")
                } catch (e: Exception) {
                    Log.e(TAG, "启用 GPU 失败，回退到 CPU: ${e.message}")
                }
            }
        }
        
        interpreter = Interpreter(modelBuffer, options)
        interpreter.allocateTensors()

        // 3. 获取输入输出形状
        val inputShape = interpreter.getInputTensor(0).shape() // [1, 640, 640, 3]
        inputHeight = inputShape[1]
        inputWidth = inputShape[2]
        
        val outputShape = interpreter.getOutputTensor(0).shape() // [1, 84, 8400]
        outputBuffer = Array(outputShape[0]) { Array(outputShape[1]) { FloatArray(outputShape[2]) } }
        outputChannel = outputShape[1]
        outputAnchors = outputShape[2]

        Log.d(TAG, "模型加载成功. 输入: $inputWidth x $inputHeight, 输出: $outputChannel channels x $outputAnchors anchors")

        // 4. 初始化预处理
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // [0, 255] -> [0, 1]
            .add(CastOp(DataType.FLOAT32))
            .build()
            
        tensorImage = TensorImage(DataType.FLOAT32)

        // 5. 尝试加载标签
        tryLoadLabels(modelBuffer)
    }

    /**
     * 尝试从 Metadata 加载标签
     */
    private fun tryLoadLabels(buffer: ByteBuffer) {
        try {
            val extractor = MetadataExtractor(buffer)
            extractor.associatedFileNames?.forEach { fileName ->
                extractor.getAssociatedFile(fileName)?.use {
                    val content = String(it.readBytes())
                    if (content.contains("names:")) {
                         val yaml = Yaml()
                         val map = yaml.load<Map<String, Any>>(content)
                         val names = map["names"] as? Map<Int, String>
                         if (names != null) {
                             labels = names.values.toList()
                             Log.d(TAG, "加载标签成功: ${labels.size} 个")
                         }
                    }
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "元数据标签加载失败: ${e.message}")
        }
    }

    /**
     * 执行检测
     * @param bitmap 输入图片
     * @return 检测结果
     */
    fun detect(bitmap: Bitmap): YOLOResult {
        val startTime = System.nanoTime()

        // 1. 预处理
        tensorImage.load(bitmap)
        val processedImage = imageProcessor.process(tensorImage)

        // 2. 推理
        interpreter.run(processedImage.buffer, outputBuffer)

        // 3. 后处理 (NMS)
        val boxes = postprocess(
            outputBuffer[0], // [84, 8400]
            bitmap.width,
            bitmap.height
        )

        val duration = (System.nanoTime() - startTime) / 1_000_000.0
        
        return YOLOResult(
            origShape = Size(bitmap.width, bitmap.height),
            boxes = boxes,
            speed = duration,
            names = labels
        )
    }

    /**
     * 后处理: 解析输出张量并执行 NMS
     * 
     * YOLOv8/v11/v26 输出格式通常为 [cx, cy, w, h, class1_conf, class2_conf, ...]
     * 张量形状为 [channels (4+nc), anchors]
     */
    private fun postprocess(output: Array<FloatArray>, imgW: Int, imgH: Int): List<Box> {
        val detectedBoxes = ArrayList<Box>()
        
        val rows = outputAnchors // 8400
        val cols = outputChannel // 84

        for (i in 0 until rows) {
            // 获取置信度最高的类别
            var maxConf = 0f
            var maxClassIndex = -1
            
            // 前4个是坐标，从第5个开始是类别置信度
            for (c in 4 until cols) {
                val conf = output[c][i]
                if (conf > maxConf) {
                    maxConf = conf
                    maxClassIndex = c - 4
                }
            }

            if (maxConf > confidenceThreshold) {
                // 解析坐标 (归一化中心点和宽高)
                val cx = output[0][i]
                val cy = output[1][i]
                val w = output[2][i]
                val h = output[3][i]

                // 转换为左上角坐标
                val x1 = cx - w / 2
                val y1 = cy - h / 2
                
                // 转换为绝对坐标
                val rect = RectF(
                    x1 * imgW,
                    y1 * imgH,
                    (x1 + w) * imgW,
                    (y1 + h) * imgH
                )
                
                // 归一化坐标记录
                val rectN = RectF(x1, y1, x1 + w, y1 + h)
                
                val labelName = if (maxClassIndex in labels.indices) labels[maxClassIndex] else "class_$maxClassIndex"
                
                detectedBoxes.add(Box(
                    index = maxClassIndex,
                    cls = labelName,
                    conf = maxConf,
                    xywh = rect,
                    xywhn = rectN
                ))
            }
        }

        // 执行 NMS (非极大值抑制)
        return nms(detectedBoxes)
    }

    /**
     * 非极大值抑制 (NMS)
     */
    private fun nms(boxes: List<Box>): List<Box> {
        val sorted = boxes.sortedByDescending { it.conf }
        val selected = ArrayList<Box>()
        
        // 简单限制数量，防止 O(N^2) 过慢
        val candidateCount = kotlin.math.min(sorted.size, numItemsThreshold * 5)
        
        val active = BooleanArray(candidateCount) { true }
        
        for (i in 0 until candidateCount) {
            if (!active[i]) continue
            
            val boxA = sorted[i]
            selected.add(boxA)
            
            if (selected.size >= numItemsThreshold) break
            
            for (j in i + 1 until candidateCount) {
                if (active[j]) {
                    val boxB = sorted[j]
                    if (iou(boxA.xywh, boxB.xywh) > iouThreshold) {
                        active[j] = false
                    }
                }
            }
        }
        
        return selected
    }

    /**
     * 计算 IoU (交并比)
     */
    private fun iou(a: RectF, b: RectF): Float {
        val left = kotlin.math.max(a.left, b.left)
        val top = kotlin.math.max(a.top, b.top)
        val right = kotlin.math.min(a.right, b.right)
        val bottom = kotlin.math.min(a.bottom, b.bottom)

        val w = kotlin.math.max(0f, right - left)
        val h = kotlin.math.max(0f, bottom - top)
        
        val inter = w * h
        val union = (a.width() * a.height()) + (b.width() * b.height()) - inter
        
        return if (union > 0) inter / union else 0f
    }

    // 设置阈值
    fun setConfidence(conf: Float) { this.confidenceThreshold = conf }
    fun setIou(iou: Float) { this.iouThreshold = iou }
}