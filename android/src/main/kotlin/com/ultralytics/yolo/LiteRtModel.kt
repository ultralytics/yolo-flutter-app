// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType

/**
 * Wraps a LiteRT 2.x [CompiledModel] behind a simple float-in / float-out API so the predictors don't deal with
 * [TensorBuffer]s or the accelerator framework directly.
 *
 * Accelerator ladder: GPU first (the CL/GL accelerator bundled with `litert`; profiled at ~4.7ms/inf for a non-end2end
 * fp16 YOLO26 on a Galaxy S26 GPU vs ~29ms CPU), falling back to CPU when the GPU can't compile the model (e.g. int8 or
 * end2end graphs). Unlike the old `Interpreter`+`GpuDelegate`, `CompiledModel` compiles the whole graph for one
 * accelerator, so a model either runs fully on GPU or fully on CPU - no per-op fragmentation.
 *
 * Input tensor name is `images` for legacy onnx2tf exports and `args_0` for litert-torch (`format=litert`) exports;
 * outputs are `output_0`, `output_1`, ... for litert-torch and `Identity`, `Identity_1`, ... for legacy onnx2tf.
 *
 * Input layout is detected from the input tensor shape: legacy onnx2tf exports are NHWC `[1,H,W,3]`, while
 * `format=litert` (litert-torch) exports are NCHW `[1,3,H,W]`. [inputDims] is always reported in the NHWC convention so
 * predictors stay layout-agnostic (matching [OrtQnnModel]); [run] feeds interleaved HWC floats, transposing them to
 * planar CHW internally for NCHW models.
 */
class LiteRtModel(
    private val context: android.content.Context,
    modelPath: String,
    useGpu: Boolean,
    private val tag: String,
) : InferenceModel {
    private data class PreparedModel(
        val model: CompiledModel,
        val inputBuffers: List<TensorBuffer>,
        val outputBuffers: List<TensorBuffer>,
        val inputDims: IntArray,
        val nchw: Boolean,
        val outputElementCounts: IntArray,
        val outputDims: List<IntArray>,
        val outputTypes: List<TensorType.ElementType?>,
    )

    private val model: CompiledModel
    private val inputBuffers: List<TensorBuffer>
    private val outputBuffers: List<TensorBuffer>
    private val outputTypes: List<TensorType.ElementType?>

    /** True when the model input is NCHW `[1,3,H,W]` (litert-torch) rather than NHWC `[1,H,W,3]` (legacy onnx2tf). */
    private val nchw: Boolean

    /** Reusable planar CHW scratch buffer for NCHW models, allocated once to avoid per-frame churn (see [run]). */
    private var chwScratch = FloatArray(0)

    /** Accelerator actually in use after the ladder resolves: "GPU" or "CPU". */
    override val accelerator: String

    /**
     * Input tensor dimensions in NHWC convention, e.g. [1, 640, 640, 3], regardless of the model's native layout (NCHW
     * litert-torch shapes are reported transposed). Model load fails if the input shape can't be read by name.
     */
    override val inputDims: IntArray

    /** Float element count of each output buffer, in order. */
    override val outputElementCounts: IntArray

    /** Output tensor dimensions, in order (e.g. [[1, 84, 8400]] for detect). Empty entries if a name doesn't resolve. */
    override val outputDims: List<IntArray>

    init {
        var prepared: PreparedModel? = null
        var acc = "CPU"
        if (useGpu) {
            try {
                prepared = prepareModel(modelPath, Accelerator.GPU)
                acc = "GPU"
            } catch (e: Throwable) {
                Log.w(tag, "GPU accelerator could not run model, falling back to CPU: ${e.message}")
            }
        }
        if (prepared == null) {
            prepared = prepareModel(modelPath, Accelerator.CPU)
            acc = "CPU"
        }
        model = prepared.model
        accelerator = acc

        inputBuffers = prepared.inputBuffers
        outputBuffers = prepared.outputBuffers
        inputDims = prepared.inputDims
        nchw = prepared.nchw
        outputElementCounts = prepared.outputElementCounts
        outputDims = prepared.outputDims
        outputTypes = prepared.outputTypes

        Log.i(
            tag,
            "LiteRT compiled on $acc; inputDims=${inputDims.toList()} " +
                "outputDims=${outputDims.map { it.toList() }} outputCounts=${outputElementCounts.toList()}",
        )
    }

    private fun prepareModel(modelPath: String, accelerator: Accelerator): PreparedModel {
        val options = CompiledModel.Options(accelerator)
        if (accelerator == Accelerator.GPU) {
            // Serialize compiled GPU programs so subsequent model opens skip CL compilation entirely.
            options.gpuOptions = CompiledModel.GpuOptions(
                serializationDir = context.codeCacheDir.absolutePath,
                modelCacheKey = "${java.io.File(modelPath).name}_${java.io.File(modelPath).length()}",
                serializeProgramCache = true,
            )
        }
        val compiled = CompiledModel.create(modelPath, options)
        val inputs: List<TensorBuffer>
        val outputs: List<TensorBuffer>
        try {
            inputs = compiled.createInputBuffers()
            outputs = compiled.createOutputBuffers()
        } catch (e: Throwable) {
            runCatching { compiled.close() }
            throw e
        }

        try {
            // Read the input shape by name. litert-torch (format=litert) names the input `args_0`; legacy onnx2tf
            // exports name it `images`. Try both and refuse to guess on a miss: silently assuming NHWC would feed an
            // NCHW graph raw HWC data (wrong detections, no crash), which is worse than failing loudly at load.
            val nativeDims = sequenceOf("images", "args_0").firstNotNullOfOrNull { name ->
                try {
                    compiled.getInputTensorType(inputName = name).layout?.dimensions?.toIntArray()?.takeIf { it.isNotEmpty() }
                } catch (e: Throwable) {
                    null
                }
            } ?: throw IllegalStateException(
                "Could not read LiteRT input tensor shape by name (tried 'images', 'args_0'); unknown input layout."
            )
            // litert-torch exports are NCHW [1,3,H,W]; legacy onnx2tf exports are NHWC [1,H,W,3]. Detect from the shape
            // and report NHWC to predictors either way so they stay layout-agnostic (run() transposes for NCHW).
            val nchw = nativeDims.size >= 4 && nativeDims[1] == 3 && nativeDims.last() != 3
            val dims = if (nchw) intArrayOf(nativeDims[0], nativeDims[2], nativeDims[3], 3) else nativeDims

            // Warm up once with a zeroed input to (a) prime the accelerator and (b) learn each output's element count,
            // which the predictors use to reshape the flat float outputs. Keep this inside the accelerator fallback
            // path: some GPU drivers compile successfully but fail on first run.
            val inputFloats = if (dims.isNotEmpty()) dims.fold(1) { a, b -> a * b } else 0
            if (inputFloats > 0) {
                inputs[0].writeFloat(FloatArray(inputFloats))
                compiled.run(inputs, outputs)
            }
            val outputTensorTypes = List(outputs.size) { i ->
                // litert-torch (format=litert) names signature outputs output_0, output_1, …; legacy onnx2tf exports
                // name them Identity, Identity_1, …. Try both so the output shape resolves for either export: a missed
                // shape leaves outputDims empty, which breaks the segment/semantic/pose predictors that read it (the
                // detect/obb/classify heads fall back to the element count, so they survive a miss).
                val legacyName = if (i == 0) "Identity" else "Identity_$i"
                sequenceOf("output_$i", legacyName).firstNotNullOfOrNull { name ->
                    try {
                        compiled.getOutputTensorType(outputName = name)
                    } catch (e: Throwable) {
                        null // also thrown for element types the Kotlin API can't read (e.g. uint8)
                    }
                }
            }
            val outputShapes = outputTensorTypes.map { it?.layout?.dimensions?.toIntArray() ?: IntArray(0) }
            val types = outputTensorTypes.map { it?.elementType }
            val elementCounts = IntArray(outputs.size) { readAsFloats(outputs[it], types[it]).size }
            return PreparedModel(compiled, inputs, outputs, dims, nchw, elementCounts, outputShapes, types)
        } catch (e: Throwable) {
            closeBuffers(inputs, outputs)
            runCatching { compiled.close() }
            throw e
        }
    }

    /**
     * Run inference: write [input] floats (interleaved HWC, as the predictors produce) into the first input buffer, run,
     * and return each output as a flat float array. NCHW models get an HWC→CHW transpose first.
     */
    override fun run(input: FloatArray): List<FloatArray> {
        val buffer = if (nchw) {
            if (chwScratch.size != input.size) chwScratch = FloatArray(input.size)
            ImageUtils.hwcToChw(input, chwScratch)
            chwScratch
        } else {
            input
        }
        inputBuffers[0].writeFloat(buffer)
        model.run(inputBuffers, outputBuffers)
        return List(outputBuffers.size) { readAsFloats(outputBuffers[it], outputTypes[it]) }
    }

    /**
     * Read a tensor buffer as floats; integer outputs (e.g. semantic class maps) are widened. Dispatch on the
     * declared element type - the native read functions don't type-check, so a mistyped read corrupts memory.
     */
    private fun readAsFloats(buffer: TensorBuffer, type: TensorType.ElementType?): FloatArray = when (type) {
        TensorType.ElementType.INT -> buffer.readInt().let { v -> FloatArray(v.size) { v[it].toFloat() } }
        TensorType.ElementType.INT8 -> widenToFloats(buffer.readInt8())
        TensorType.ElementType.INT64 -> buffer.readLong().let { v -> FloatArray(v.size) { v[it].toFloat() } }
        else -> buffer.readFloat() // FLOAT, or null when the type can't be read (all official assets are float)
    }

    override fun close() {
        closeBuffers(inputBuffers, outputBuffers)
        try {
            model.close()
        } catch (_: Throwable) {
            // best-effort
        }
    }

    private fun closeBuffers(inputs: List<TensorBuffer>, outputs: List<TensorBuffer>) {
        for (buffer in inputs) {
            try {
                buffer.close()
            } catch (_: Throwable) {
                // best-effort
            }
        }
        for (buffer in outputs) {
            try {
                buffer.close()
            } catch (_: Throwable) {
                // best-effort
            }
        }
    }
}
