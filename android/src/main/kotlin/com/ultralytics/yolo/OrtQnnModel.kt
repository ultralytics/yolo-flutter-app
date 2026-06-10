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
    private val outputNames: List<String>
    private val inputShape: LongArray
    private val nhwcInput: Boolean
    private val nchw: FloatArray
    private val outputs: List<FloatArray>

    override val accelerator = "NPU"
    override val inputDims: IntArray
    override val outputDims: List<IntArray>
    override val outputElementCounts: IntArray

    init {
        // Point the Hexagon DSP loader at the APK's own QAIRT Skel libraries; otherwise fastrpc falls back to
        // /vendor/lib/rfsa/adsp and pairs the in-process Stub with whatever QAIRT vintage the device vendor shipped,
        // which fails QnnDevice_create with INVALID_CONFIG on version mismatches.
        try {
            android.system.Os.setenv(
                "ADSP_LIBRARY_PATH",
                context.applicationInfo.nativeLibraryDir + ";/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp",
                true,
            )
        } catch (e: Throwable) {
            Log.w(tag, "Could not set ADSP_LIBRARY_PATH: ${e.message}")
        }
        session = OrtSession.SessionOptions().use { options ->
            options.addQnn(mapOf("backend_path" to "libQnnHtp.so", "htp_performance_mode" to "burst"))
            createSession(context, modelPath, options)
        }
        try {
            inputName = session.inputNames.first()
            inputShape = (session.inputInfo.getValue(inputName).info as TensorInfo).shape
            // Channel-last QNN exports take [1, H, W, 3] directly (no CPU transpose); legacy exports are NCHW
            nhwcInput = inputShape.size == 4 && inputShape[3] == 3L && inputShape[1] != 3L
            require(inputShape.size == 4 && (nhwcInput || inputShape[1] == 3L)) {
                "Expected a [1, 3, H, W] or [1, H, W, 3] input, got ${inputShape.toList()} for '$inputName'"
            }
            val height = (if (nhwcInput) inputShape[1] else inputShape[2]).toInt()
            val width = (if (nhwcInput) inputShape[2] else inputShape[3]).toInt()
            inputDims = intArrayOf(1, height, width, 3)
            nchw = if (nhwcInput) FloatArray(0) else FloatArray(3 * height * width)

            // One ordered name list drives both shape discovery and result reads, so they can never desynchronize
            outputNames = session.outputNames.toList()
            outputDims = outputNames.map { name ->
                (session.outputInfo.getValue(name).info as TensorInfo).shape.map(Long::toInt).toIntArray()
            }
            outputElementCounts = IntArray(outputDims.size) { i -> outputDims[i].fold(1) { a, b -> a * b } }
            // Reused per-run output buffers: semantic logits alone are ~80MB at 1024px, so allocating fresh
            // arrays every predict churns the Java heap into OOM on sustained inference
            outputs = outputElementCounts.map { FloatArray(it) }
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

    /** Run inference on NHWC interleaved-RGB floats, returning each output as a flat float array. */
    override fun run(input: FloatArray): List<FloatArray> {
        val floats = if (nhwcInput) {
            input // channel-last graph: feed the predictors' NHWC buffer directly
        } else {
            val hw = nchw.size / 3
            for (i in 0 until hw) {
                val j = i * 3
                nchw[i] = input[j]
                nchw[hw + i] = input[j + 1]
                nchw[2 * hw + i] = input[j + 2]
            }
            nchw
        }
        OnnxTensor.createTensor(env, FloatBuffer.wrap(floats), inputShape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { results ->
                return outputNames.mapIndexed { i, name ->
                    readOutput(results.get(name).get() as OnnxTensor, outputs[i])
                }
            }
        }
    }

    /** Copy a result tensor into [target] as floats; uint8 outputs (e.g. semantic class maps) are widened. */
    private fun readOutput(tensor: OnnxTensor, target: FloatArray): FloatArray {
        val info = tensor.info
        if (info.type == ai.onnxruntime.OnnxJavaType.UINT8 || info.type == ai.onnxruntime.OnnxJavaType.INT8) {
            val bytes = tensor.byteBuffer
            val count = bytes.remaining()
            for (i in 0 until count) {
                target[i] = (bytes.get(i).toInt() and 0xFF).toFloat()
            }
        } else {
            val buffer = tensor.floatBuffer
            buffer.get(target, 0, buffer.remaining())
        }
        return target
    }

    override fun close() {
        try {
            session.close()
        } catch (_: Throwable) {
            // best-effort
        }
    }
}
