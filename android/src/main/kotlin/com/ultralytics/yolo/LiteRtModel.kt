// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * Wraps a LiteRT 2.x [CompiledModel] behind a simple float-in / float-out API so the predictors don't deal with
 * [TensorBuffer]s or the accelerator framework directly.
 *
 * Accelerator ladder: GPU first (the CL/GL accelerator bundled with `litert`; profiled at ~4.7ms/inf for a non-end2end
 * fp16 YOLO26 on a Galaxy S26 GPU vs ~29ms CPU), falling back to CPU when the GPU can't compile the model (e.g. int8 or
 * end2end graphs). Unlike the old `Interpreter`+`GpuDelegate`, `CompiledModel` compiles the whole graph for one
 * accelerator, so a model either runs fully on GPU or fully on CPU - no per-op fragmentation.
 *
 * Tensor names follow the Ultralytics tflite export convention: input `images`, outputs `Identity`, `Identity_1`, ...
 */
class LiteRtModel(modelPath: String, useGpu: Boolean, private val tag: String) {
    private val model: CompiledModel
    private val inputBuffers: List<TensorBuffer>
    private val outputBuffers: List<TensorBuffer>

    /** Accelerator actually in use after the ladder resolves: "GPU" or "CPU". */
    val accelerator: String

    /** Input tensor dimensions, e.g. [1, 640, 640, 3]. Empty if the model doesn't use the conventional `images` name. */
    val inputDims: IntArray

    /** Float element count of each output buffer, in order. */
    val outputElementCounts: IntArray

    /** Output tensor dimensions, in order (e.g. [[1, 84, 8400]] for detect). Empty entries if a name doesn't resolve. */
    val outputDims: List<IntArray>

    init {
        var compiled: CompiledModel? = null
        var acc = "CPU"
        if (useGpu) {
            try {
                compiled = CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.GPU))
                acc = "GPU"
            } catch (e: Throwable) {
                Log.w(tag, "GPU accelerator could not compile model, falling back to CPU: ${e.message}")
            }
        }
        if (compiled == null) {
            compiled = CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.CPU))
            acc = "CPU"
        }
        model = compiled
        accelerator = acc

        inputBuffers = model.createInputBuffers()
        outputBuffers = model.createOutputBuffers()

        inputDims = try {
            model.getInputTensorType(inputName = "images").layout?.dimensions?.toIntArray() ?: IntArray(0)
        } catch (e: Throwable) {
            Log.w(tag, "Could not read input tensor type: ${e.message}")
            IntArray(0)
        }

        // Warm up once with a zeroed input to (a) prime the accelerator and (b) learn each output's element count,
        // which the predictors use to reshape the flat float outputs.
        val inputFloats = if (inputDims.isNotEmpty()) inputDims.fold(1) { a, b -> a * b } else 0
        if (inputFloats > 0) {
            inputBuffers[0].writeFloat(FloatArray(inputFloats))
            model.run(inputBuffers, outputBuffers)
        }
        outputElementCounts = IntArray(outputBuffers.size) { outputBuffers[it].readFloat().size }
        outputDims = List(outputBuffers.size) { i ->
            val name = if (i == 0) "Identity" else "Identity_$i"
            try {
                model.getOutputTensorType(outputName = name).layout?.dimensions?.toIntArray() ?: IntArray(0)
            } catch (e: Throwable) {
                IntArray(0)
            }
        }
        Log.i(
            tag,
            "LiteRT compiled on $acc; inputDims=${inputDims.toList()} " +
                "outputDims=${outputDims.map { it.toList() }} outputCounts=${outputElementCounts.toList()}",
        )
    }

    /** Run inference: write [input] floats into the first input buffer, run, and return each output as a flat float array. */
    fun run(input: FloatArray): List<FloatArray> {
        inputBuffers[0].writeFloat(input)
        model.run(inputBuffers, outputBuffers)
        return List(outputBuffers.size) { outputBuffers[it].readFloat() }
    }

    fun close() {
        try {
            model.close()
        } catch (_: Throwable) {
            // best-effort
        }
    }
}
