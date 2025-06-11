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
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import kotlin.math.max
import kotlin.math.min

class Segmenter(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
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

    // TFLite Interpreter options
    private val interpreterOptions = (customOptions ?: Interpreter.Options()).apply {
        // If no custom options provided, use default threads
        if (customOptions == null) {
            setNumThreads(Runtime.getRuntime().availableProcessors())
        }
        
        if (useGpu) {
            try {
                addDelegate(GpuDelegate())
                Log.d("Segmenter", "GPU delegate is used.")
            } catch (e: Exception) {
                Log.e("Segmenter", "GPU delegate error: ${e.message}")
            }
        }
    }

    /** ImageProcessor for image preprocessing - separate ones for camera portrait/landscape and single images */
    private lateinit var imageProcessorCameraPortrait: ImageProcessor
    private lateinit var imageProcessorCameraLandscape: ImageProcessor
    private lateinit var imageProcessorSingleImage: ImageProcessor
    
    // Reuse ByteBuffer for input to reduce allocations
    private lateinit var inputBuffer: ByteBuffer
    
    // Reuse output arrays to reduce allocations
    private lateinit var output0: Array<Array<FloatArray>>
    private lateinit var output1: Array<Array<Array<FloatArray>>>

    init {
        // Load model file (automatic extension appending)
        val modelBuffer = YOLOUtils.loadModelFile(context, modelPath)

        // ===== Load label information (try Appended ZIP → FlatBuffers in order) =====
        var loadedLabels = YOLOFileUtils.loadLabelsFromAppendedZip(context, modelPath)
        var labelsWereLoaded = loadedLabels != null

        if (labelsWereLoaded) {
            this.labels = loadedLabels!! // Use labels from appended ZIP
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
            Log.w("Segmenter", "No embedded labels found from appended ZIP or FlatBuffers. Using labels passed via constructor (if any) or an empty list.")
            if (this.labels.isEmpty()) {
                Log.w("Segmenter", "Warning: No labels loaded and no labels provided via constructor. Detections might lack class names.")
            }
        }

        // Create Interpreter
        interpreter = Interpreter(modelBuffer, interpreterOptions)
        // Call allocateTensors() once during initialization
        interpreter.allocateTensors()
        Log.d("Segmenter", "TFLite model loaded and tensors allocated")

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
        
        // Initialize output0 buffer (example: [1,116,2100])
        val batch0 = out0Shape[0]
        out0NumFeatures = out0Shape[1]
        out0NumAnchors = out0Shape[2]
        output0 = Array(batch0) { Array(out0NumFeatures) { FloatArray(out0NumAnchors) } }
        
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

        // Initialize ImageProcessor - separate ones for camera portrait/landscape and single images
        
        // 1. For camera feed in portrait mode (with rotation)
        imageProcessorCameraPortrait = ImageProcessor.Builder()
            .add(Rot90Op(3)) // 270-degree rotation
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .add(CastOp(DataType.FLOAT32))
            .build()
            
        // 2. For camera feed in landscape mode (no rotation)
        imageProcessorCameraLandscape = ImageProcessor.Builder()
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .add(CastOp(DataType.FLOAT32))
            .build()
            
        // 3. For single images (no rotation)
        imageProcessorSingleImage = ImageProcessor.Builder()
            .add(ResizeOp(inHeight, inWidth, ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .add(CastOp(DataType.FLOAT32))
            .build()
    }

    override fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean, isLandscape: Boolean): YOLOResult {
        t0 = System.nanoTime()

        // (1) Preprocess with TensorImage
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        
        // Choose appropriate processor based on input source and orientation
        val processedImage = if (rotateForCamera) {
            // Use appropriate camera processor based on device orientation
            if (isLandscape) {
                imageProcessorCameraLandscape.process(tensorImage)
            } else {
                imageProcessorCameraPortrait.process(tensorImage)
            }
        } else {
            // Use single image processor (no rotation) for regular images
            imageProcessorSingleImage.process(tensorImage)
        }
        
        // Reuse the pre-allocated input buffer
        inputBuffer.clear()
        inputBuffer.put(processedImage.buffer)
        inputBuffer.rewind()

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

        val boxes = mutableListOf<Box>()
        for ((normRect, cls, score, maskCoeffs) in rawDetections) {
            // normRect already contains normalized coordinates (0-1)
            
            // Convert to absolute pixel coordinates for xywh
            val rectF = RectF(
                normRect.left * origWidth,
                normRect.top * origHeight,
                normRect.right * origWidth,
                normRect.bottom * origHeight
            )
            
            val label = labels.getOrElse(cls) { "Unknown" }
            // Use normRect for xywhn (normalized 0-1 coordinates) and rectF for xywh (pixel coordinates)
            boxes.add(Box(cls, label, score, rectF, normRect))
        }

        val (combinedMask, probMasks) = generateCombinedMaskImage(
            detections = rawDetections,
            protos = output1[0],
            maskW = maskW,
            maskH = maskH,
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
        threshold: Float
    ): Pair<Bitmap?, List<List<List<Float>>>?> {
        if (detections.isEmpty()) return Pair(null, null)
        val combinedPixels = IntArray(maskW * maskH) { Color.TRANSPARENT }
        val probabilityMasks = mutableListOf<List<List<Float>>>()
        detections.forEachIndexed { detIndex, det ->
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
            probabilityMasks.add(pm.map { it.toList() })
        }
        val bmp = Bitmap.createBitmap(maskW, maskH, Bitmap.Config.ARGB_8888)
        bmp.setPixels(combinedPixels, 0, maskW, 0, 0, maskW, maskH)
        return Pair(bmp, probabilityMasks)
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
                Log.d("Segmenter", "Found associated file: $fileName")
                extractor.getAssociatedFile(fileName)?.use { stream ->
                    val fileString = String(stream.readBytes(), Charsets.UTF_8)
                    Log.d("Segmenter", "Associated file contents:\n$fileString")

                    val yaml = Yaml()
                    @Suppress("UNCHECKED_CAST")
                    val data = yaml.load<Map<String, Any>>(fileString)
                    if (data != null && data.containsKey("names")) {
                        val namesMap = data["names"] as? Map<Int, String>
                        if (namesMap != null) {
                            labels = namesMap.values.toList()
                            Log.d("Segmenter", "Loaded labels from metadata: $labels")
                            return true
                        }
                    }
                }
            }
        } else {
            Log.d("Segmenter", "No associated files found in the metadata.")
        }
        false
    } catch (e: Exception) {
        Log.e("Segmenter", "Failed to extract metadata: ${e.message}")
        false
    }
}