// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.util.Log
import java.io.File
import java.io.IOException
import java.nio.FloatBuffer

/**
 * Runs a Qualcomm QNN context-binary ONNX model (an Ultralytics `*_qnn.onnx` export) on the Snapdragon Hexagon NPU
 * via the ONNX Runtime QNN Execution Provider.
 *
 * `onnxruntime-android-qnn` is a compileOnly dependency of the plugin so consumers don't pay its size by default;
 * apps opt in by adding it to their own build.gradle. The context binary is precompiled for a Hexagon HTP
 * architecture at export time, so the session loads without on-device graph compilation - and without a CPU
 * fallback: on non-Snapdragon hardware session creation throws and callers should fall back to a TFLite model.
 *
 * Predictors supply NHWC interleaved-RGB floats (the TFLite layout); this wrapper transposes into the NCHW planar
 * layout ONNX models expect and reports [inputDims] in the NHWC convention so predictors stay runtime-agnostic.
 */
class OrtQnnModel(context: Context, modelPath: String, private val tag: String) : InferenceModel {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val nchwShape: LongArray
    private val nchw: FloatArray

    override val accelerator = "NPU"
    override val inputDims: IntArray
    override val outputDims: List<IntArray>
    override val outputElementCounts: IntArray

    init {
        session = OrtSession.SessionOptions().use { options ->
            options.addQnn(mapOf("backend_path" to "libQnnHtp.so", "htp_performance_mode" to "burst"))
            createSession(context, modelPath, options)
        }
        try {
            inputName = session.inputNames.first()
            nchwShape = (session.inputInfo.getValue(inputName).info as TensorInfo).shape // [1, 3, H, W]
            require(nchwShape.size == 4 && nchwShape[1] == 3L) {
                "Expected an NCHW [1, 3, H, W] input, got ${nchwShape.toList()} for '$inputName'"
            }
            val height = nchwShape[2].toInt()
            val width = nchwShape[3].toInt()
            inputDims = intArrayOf(1, height, width, 3)
            nchw = FloatArray(3 * height * width)

            outputDims = session.outputNames.map { name ->
                (session.outputInfo.getValue(name).info as TensorInfo).shape.map(Long::toInt).toIntArray()
            }
            outputElementCounts = IntArray(outputDims.size) { i -> outputDims[i].fold(1) { a, b -> a * b } }
        } catch (t: Throwable) {
            close()
            throw t
        }

        Log.i(
            tag,
            "ONNX Runtime QNN session on NPU; inputDims=${inputDims.toList()} " +
                "outputDims=${outputDims.map { it.toList() }}",
        )
    }

    private fun createSession(context: Context, modelPath: String, options: OrtSession.SessionOptions): OrtSession {
        val file = File(modelPath)
        if (file.isAbsolute && file.exists()) return env.createSession(modelPath, options)
        // Flutter asset: EPContext models are self-contained, so load the bytes through the AssetManager
        for (path in listOf(modelPath, modelPath.removePrefix("flutter_assets/"), "flutter_assets/$modelPath").distinct()) {
            try {
                context.assets.open(path).use { return env.createSession(it.readBytes(), options) }
            } catch (_: IOException) {
                // try the next candidate path
            }
        }
        throw IllegalArgumentException("QNN model not found at '$modelPath' (filesystem or assets)")
    }

    /** Run inference: transpose [input] from NHWC to NCHW, run, and return each output as a flat float array. */
    override fun run(input: FloatArray): List<FloatArray> {
        val hw = nchw.size / 3
        for (i in 0 until hw) {
            val j = i * 3
            nchw[i] = input[j]
            nchw[hw + i] = input[j + 1]
            nchw[2 * hw + i] = input[j + 2]
        }
        OnnxTensor.createTensor(env, FloatBuffer.wrap(nchw), nchwShape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { results ->
                return List(results.size()) { i ->
                    val buffer = (results.get(i) as OnnxTensor).floatBuffer
                    FloatArray(buffer.remaining()).also { buffer.get(it) }
                }
            }
        }
    }

    override fun close() {
        try {
            session.close()
        } catch (_: Throwable) {
            // best-effort
        }
    }
}
