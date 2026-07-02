// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.graphics.*
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object ImageUtils {
    // Shared read-only paint for frame preprocessing.
    private val filterPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val blackPaint = Paint().apply { color = Color.BLACK }

    data class LetterboxTransform(
        val gain: Float,
        val padX: Float,
        val padY: Float,
        val padRight: Float,
        val padBottom: Float,
        val resizedWidth: Int,
        val resizedHeight: Int
    )

    @JvmStatic
    fun letterboxTransform(
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int,
        centerCrop: Boolean = false
    ): LetterboxTransform? {
        if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) return null

        val scaleX = targetWidth.toFloat() / sourceWidth
        val scaleY = targetHeight.toFloat() / sourceHeight
        val gain = if (centerCrop) max(scaleX, scaleY) else min(scaleX, scaleY)
        if (gain <= 0f) return null
        val resizedWidth = (sourceWidth * gain).roundToInt()
        val resizedHeight = (sourceHeight * gain).roundToInt()
        val padWidth = targetWidth - resizedWidth
        val padHeight = targetHeight - resizedHeight
        // Match Ultralytics LetterBox leading/trailing pad rounding.
        val padX = (padWidth / 2f - 0.1f).roundToInt().toFloat()
        val padY = (padHeight / 2f - 0.1f).roundToInt().toFloat()
        val padRight = (padWidth / 2f + 0.1f).roundToInt().toFloat()
        val padBottom = (padHeight / 2f + 0.1f).roundToInt().toFloat()
        return LetterboxTransform(gain, padX, padY, padRight, padBottom, resizedWidth, resizedHeight)
    }

    /**
     * Sample to convert ImageProxy to NV21 (BYTE array), then [YuvImage] -> [Bitmap]
     */
    @JvmStatic
    fun toBitmap(imageProxy: ImageProxy): Bitmap? {
        // Fast path: CameraX is configured for OUTPUT_IMAGE_FORMAT_RGBA_8888, so the frame is a single RGBA plane we
        // can copy straight into a Bitmap (~2-5ms). This replaces a YUV->NV21->JPEG-encode@100->JPEG-decode round-trip
        // that cost ~100ms/frame (~5 FPS).
        if (imageProxy.format == PixelFormat.RGBA_8888 && imageProxy.planes.size == 1) {
            val plane = imageProxy.planes[0]
            val pixelStride = plane.pixelStride
            val rowStride = plane.rowStride
            val rowPadding = rowStride - pixelStride * imageProxy.width
            // When rowStride has padding the buffer is wider than the image; copy at full stride width then crop back.
            val paddedWidth = imageProxy.width + rowPadding / pixelStride
            val bitmap = Bitmap.createBitmap(paddedWidth, imageProxy.height, Bitmap.Config.ARGB_8888)
            plane.buffer.rewind()
            bitmap.copyPixelsFromBuffer(plane.buffer)
            return if (rowPadding == 0) {
                bitmap
            } else {
                Bitmap.createBitmap(bitmap, 0, 0, imageProxy.width, imageProxy.height)
            }
        }

        // Fallback for YUV_420_888 frames (older config / devices that don't honor the RGBA request).
        val nv21 = yuv420888ToNv21(imageProxy)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        return yuvImageToBitmap(yuvImage)
    }

    private fun yuvImageToBitmap(yuvImage: YuvImage): Bitmap? {
        val out = ByteArrayOutputStream()
        val success = yuvImage.compressToJpeg(
            Rect(0, 0, yuvImage.width, yuvImage.height),
            100,
            out
        )
        if (!success) return null
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }


    private fun yuv420888ToNv21(imageProxy: ImageProxy): ByteArray {
        val cropRect = imageProxy.cropRect
        val pixelCount = cropRect.width() * cropRect.height()
        val pixelSizeBits = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888)
        val outputBuffer = ByteArray(pixelCount * pixelSizeBits / 8)
        imageToByteBuffer(imageProxy, outputBuffer, pixelCount)
        return outputBuffer
    }


    private fun imageToByteBuffer(
        imageProxy: ImageProxy,
        outputBuffer: ByteArray,
        pixelCount: Int
    ) {
        require(imageProxy.format == ImageFormat.YUV_420_888) {
            "Input ImageProxy must be in YUV_420_888 format."
        }

        val imageCrop = imageProxy.cropRect
        val imagePlanes = imageProxy.planes

        for (planeIndex in imagePlanes.indices) {
            val (outputStride, startOffset) = when (planeIndex) {
                0 -> Pair(1, 0)               // Y
                1 -> Pair(2, pixelCount + 1)  // U
                2 -> Pair(2, pixelCount)      // V
                else -> return
            }

            val plane = imagePlanes[planeIndex]
            val planeBuffer: ByteBuffer = plane.buffer
            val rowStride = plane.rowStride
            val pixelStride = plane.pixelStride

            val planeCrop = if (planeIndex == 0) {
                imageCrop
            } else {
                Rect(
                    imageCrop.left / 2,
                    imageCrop.top / 2,
                    imageCrop.right / 2,
                    imageCrop.bottom / 2
                )
            }

            val planeWidth = planeCrop.width()
            val planeHeight = planeCrop.height()

            val rowBuffer = ByteArray(rowStride)
            var outputOffset = startOffset

            val rowLength = if (pixelStride == 1 && outputStride == 1) {
                planeWidth
            } else {
                (planeWidth - 1) * pixelStride + 1
            }

            for (row in 0 until planeHeight) {
                planeBuffer.position(
                    (row + planeCrop.top) * rowStride +
                            planeCrop.left * pixelStride
                )

                if (pixelStride == 1 && outputStride == 1) {
                    planeBuffer.get(outputBuffer, outputOffset, rowLength)
                    outputOffset += rowLength
                } else {
                    planeBuffer.get(rowBuffer, 0, rowLength)
                    for (col in 0 until planeWidth) {
                        outputBuffer[outputOffset] = rowBuffer[col * pixelStride]
                        outputOffset += outputStride
                    }
                }
            }
        }
    }

    @JvmStatic
    fun prepareBitmapForModel(
        bitmap: Bitmap,
        targetBitmap: Bitmap,
        rotateForCamera: Boolean,
        isLandscape: Boolean,
        isFrontCamera: Boolean,
        rotationDegrees: Int? = null,
        centerCrop: Boolean = false
    ): Bitmap {
        val degrees = cameraRotationDegrees(rotateForCamera, isLandscape, isFrontCamera, rotationDegrees)
        val isRotated = degrees % 180 != 0
        val orientedWidth = if (isRotated) bitmap.height else bitmap.width
        val orientedHeight = if (isRotated) bitmap.width else bitmap.height
        val targetWidth = targetBitmap.width
        val targetHeight = targetBitmap.height
        val transform = letterboxTransform(orientedWidth, orientedHeight, targetWidth, targetHeight, centerCrop)
            ?: return targetBitmap

        Canvas(targetBitmap).apply {
            clearLetterboxPadding(transform, targetWidth, targetHeight)
            save()
            translate(transform.padX + transform.resizedWidth / 2f, transform.padY + transform.resizedHeight / 2f)
            rotate(degrees.toFloat())
            scale(transform.gain, transform.gain)
            drawBitmap(bitmap, -bitmap.width / 2f, -bitmap.height / 2f, filterPaint)
            restore()
        }
        return targetBitmap
    }

    private fun Canvas.clearLetterboxPadding(
        transform: LetterboxTransform,
        targetWidth: Int,
        targetHeight: Int
    ) {
        val left = transform.padX.coerceAtLeast(0f)
        val top = transform.padY.coerceAtLeast(0f)
        val right = (targetWidth - transform.padRight).coerceAtMost(targetWidth.toFloat())
        val bottom = (targetHeight - transform.padBottom).coerceAtMost(targetHeight.toFloat())
        if (left > 0f) drawRect(0f, 0f, left, targetHeight.toFloat(), blackPaint)
        if (right < targetWidth) drawRect(right, 0f, targetWidth.toFloat(), targetHeight.toFloat(), blackPaint)
        if (top > 0f) drawRect(left, 0f, right, top, blackPaint)
        if (bottom < targetHeight) drawRect(left, bottom, right, targetHeight.toFloat(), blackPaint)
    }

    private fun cameraRotationDegrees(
        rotateForCamera: Boolean,
        isLandscape: Boolean,
        isFrontCamera: Boolean,
        rotationDegrees: Int?
    ): Int {
        if (!rotateForCamera) return 0

        val fallbackDegrees = if (isLandscape) 0 else if (isFrontCamera) 90 else 270
        return (rotationDegrees ?: fallbackDegrees).floorMod(360)
    }

    private fun Int.floorMod(other: Int): Int = ((this % other) + other) % other

    @JvmStatic
    fun copyRgbBitmapToFloatBuffer(
        bitmap: Bitmap,
        byteBuffer: ByteBuffer,
        pixels: IntArray,
        inputMean: Float = 0f,
        inputStd: Float = 255f
    ) {
        byteBuffer.clear()
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val pixelCount = bitmap.width * bitmap.height
        for (i in 0 until pixelCount) {
            val pixel = pixels[i]
            byteBuffer.putFloat((((pixel shr 16) and 0xFF) - inputMean) / inputStd)
            byteBuffer.putFloat((((pixel shr 8) and 0xFF) - inputMean) / inputStd)
            byteBuffer.putFloat(((pixel and 0xFF) - inputMean) / inputStd)
        }
        byteBuffer.rewind()
    }

    // FloatArray variant for the LiteRT 2.x CompiledModel path (TensorBuffer.writeFloat takes a float[], not a
    // ByteBuffer). Writes interleaved HWC or planar CHW RGB, normalized to [0,1] by default.
    @JvmStatic
    fun copyRgbBitmapToFloatArray(
        bitmap: Bitmap,
        out: FloatArray,
        pixels: IntArray,
        inputMean: Float = 0f,
        inputStd: Float = 255f,
        channelsFirst: Boolean = false
    ) {
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val pixelCount = bitmap.width * bitmap.height
        val invStd = 1f / inputStd
        if (channelsFirst) {
            val plane = pixelCount
            var r = 0
            var g = plane
            var b = plane * 2
            for (i in 0 until pixelCount) {
                val pixel = pixels[i]
                out[r++] = (((pixel shr 16) and 0xFF) - inputMean) * invStd
                out[g++] = (((pixel shr 8) and 0xFF) - inputMean) * invStd
                out[b++] = ((pixel and 0xFF) - inputMean) * invStd
            }
        } else {
            var j = 0
            for (i in 0 until pixelCount) {
                val pixel = pixels[i]
                out[j++] = (((pixel shr 16) and 0xFF) - inputMean) * invStd
                out[j++] = (((pixel shr 8) and 0xFF) - inputMean) * invStd
                out[j++] = ((pixel and 0xFF) - inputMean) * invStd
            }
        }
    }

    /**
     * Process grayscale image for 1-channel classification models
     * Optimized for handwriting recognition (EMNIST-like models)
     * 
     * @param bitmap Input bitmap to process
     * @param targetWidth Target width for the model
     * @param targetHeight Target height for the model  
     * @param outputBuffer Reusable buffer for 1-channel float32 data
     * @param pixels Reusable pixel scratch array
     * @param enableColorInversion Whether to invert colors (white-on-black → black-on-white)
     * @param enableMaxNormalization Whether to use 0-1 normalization instead of mean/std
     * @param inputMean Mean value for normalization
     * @param inputStd Standard deviation for normalization
     * @return ByteBuffer ready for TensorFlow Lite inference
     */
    @JvmStatic
    fun processGrayscaleImage(
        bitmap: Bitmap,
        targetWidth: Int,
        targetHeight: Int,
        outputBuffer: ByteBuffer,
        pixels: IntArray,
        enableColorInversion: Boolean = false,
        enableMaxNormalization: Boolean = false,
        inputMean: Float = 0f,
        inputStd: Float = 255f
    ): ByteBuffer {
        val scaledBitmap = if (bitmap.width == targetWidth && bitmap.height == targetHeight) {
            bitmap
        } else {
            val targetBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
            prepareBitmapForModel(
                bitmap = bitmap,
                targetBitmap = targetBitmap,
                rotateForCamera = false,
                isLandscape = false,
                isFrontCamera = false,
                centerCrop = true
            )
        }
        
        outputBuffer.clear()
        
        // Process each pixel
        scaledBitmap.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)
        
        for (i in 0 until targetWidth * targetHeight) {
            val pixel = pixels[i]
            // Extract RGB components
            val r = (pixel shr 16) and 0xFF
            val g = (pixel shr 8) and 0xFF  
            val b = pixel and 0xFF
            
            // Convert to grayscale using luminance formula
            var gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f
            
            // Apply color inversion if enabled (for white-on-black handwriting)
            if (enableColorInversion) {
                gray = 1.0f - gray
            }
            
            // Apply normalization based on options
            val normalizedValue = if (enableMaxNormalization) {
                // Simple 0-1 normalization (already done above)
                gray
            } else {
                // Standard normalization using mean/std
                (gray - inputMean) / inputStd
            }
            
            outputBuffer.putFloat(normalizedValue)
        }
        
        // Clean up scaled bitmap if it's different from input
        if (scaledBitmap !== bitmap) {
            scaledBitmap.recycle()
        }
        
        outputBuffer.rewind()
        return outputBuffer
    }


}
