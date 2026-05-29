// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.util.Log
import java.nio.MappedByteBuffer
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate

/**
 * Builds a TFLite [Interpreter] on the fastest available path, falling back when a delegate can't load the model:
 *   1. NNAPI — routes to the device's neural accelerator (NPU/DSP, GPU otherwise). Best for the int8-quantized YOLO26
 *      models; the GPU delegate alone rejects ops like int8 CONCATENATION and throws at interpreter creation.
 *   2. GPU delegate — via [gpuOptions] (the caller's options, which already carry a GpuDelegate when useGpu is set).
 *   3. CPU (XNNPACK) — always works.
 *
 * Without this fallback chain a delegate that can't compile the model crashed the whole model load, leaving the camera
 * with no predictor and therefore no detections.
 */
fun createInterpreterFastestFirst(
    modelBuffer: MappedByteBuffer,
    useGpu: Boolean,
    gpuOptions: Interpreter.Options,
    tag: String,
): Interpreter {
    // useGpu == false: honor the caller's options as-is (no acceleration requested / custom options).
    if (!useGpu) return Interpreter(modelBuffer, gpuOptions)

    val threads = Runtime.getRuntime().availableProcessors()

    // 1) NNAPI (neural accelerator).
    try {
        return Interpreter(
            modelBuffer,
            Interpreter.Options().apply {
                setNumThreads(threads)
                addDelegate(NnApiDelegate())
            },
        )
    } catch (e: Throwable) {
        Log.w(tag, "NNAPI delegate failed, trying GPU: ${e.message}")
    }

    // 2) GPU delegate (caller's gpuOptions already include a GpuDelegate).
    try {
        return Interpreter(modelBuffer, gpuOptions)
    } catch (e: Throwable) {
        Log.w(tag, "GPU delegate failed, falling back to CPU: ${e.message}")
    }

    // 3) CPU.
    return Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(threads) })
}
