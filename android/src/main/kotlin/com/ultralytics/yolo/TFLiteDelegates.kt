// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.util.Log
import java.nio.MappedByteBuffer
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate

/**
 * Builds a TFLite [Interpreter] on the fastest available path for the model, falling back when a delegate can't load
 * it:
 *   1. GPU delegate — fastest for float/fp16 models. It rejects ops it can't accelerate (e.g. the int8 CONCATENATION
 *      in YOLO26 int8 models) and throws at interpreter creation; we then close it and fall through to CPU. Closing is
 *      important: a created-but-unapplied GpuDelegate keeps a GL context/worker thread alive that steals a CPU core
 *      and measurably slowed both preprocessing and inference (~2.6x) until released.
 *   2. XNNPACK CPU — TFLite's default CPU backend ([cpuOptions]), with optimized int8/float ARM kernels (i8mm/dotprod).
 *
 * NNAPI is deliberately NOT used. On a Galaxy S26 (Android 16) routing the int8 YOLO26 model through the NNAPI
 * delegate measured ~103ms/inference vs ~44ms on XNNPACK CPU - 2.3x slower. Google deprecated the NNAPI system API
 * (Android 15+) and vendors stopped maintaining its drivers, so it commonly falls back to a slow reference path while
 * still reporting "delegate created". For genuine NPU acceleration the path forward is a vendor delegate (Qualcomm
 * QNN / Samsung ENN) or LiteRT's newer accelerator APIs - tracked separately, not NNAPI.
 *
 * The GPU delegate is created here (not baked into [cpuOptions] by the caller) precisely so a failed attempt can be
 * closed. Without this fallback chain a delegate that can't compile the model crashed the whole model load, leaving
 * the camera with no predictor and therefore no detections on Android.
 */
fun createInterpreterFastestFirst(
    modelBuffer: MappedByteBuffer,
    useGpu: Boolean,
    cpuOptions: Interpreter.Options,
    tag: String,
): Interpreter {
    // 1) GPU delegate.
    if (useGpu) {
        val gpuDelegate = try {
            GpuDelegate()
        } catch (e: Throwable) {
            Log.w(tag, "GPU delegate unavailable: ${e.message}")
            null
        }
        if (gpuDelegate != null) {
            try {
                return Interpreter(
                    modelBuffer,
                    Interpreter.Options().apply {
                        setNumThreads(Runtime.getRuntime().availableProcessors())
                        addDelegate(gpuDelegate)
                    },
                )
            } catch (e: Throwable) {
                Log.w(tag, "GPU delegate failed, using XNNPACK CPU: ${e.message}")
                try {
                    gpuDelegate.close()
                } catch (_: Throwable) {
                    // best-effort release
                }
            }
        }
    }

    // 2) XNNPACK CPU (caller's options, which carry thread count / custom options but no GPU delegate).
    return Interpreter(modelBuffer, cpuOptions)
}
