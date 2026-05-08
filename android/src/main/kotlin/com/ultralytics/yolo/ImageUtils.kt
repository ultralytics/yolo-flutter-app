// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.graphics.*
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object ImageUtils {
    private val filterPaint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.DITHER_FLAG)

    /**
     * Sample to convert ImageProxy to NV21 (BYTE array), then [YuvImage] -> [Bitmap]
     */
    @JvmStatic
    fun toBitmap(imageProxy: ImageProxy): Bitmap? {
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
        val orientedBitmap = orientBitmapForCamera(
            bitmap,
            rotateForCamera,
            isLandscape,
            isFrontCamera,
            rotationDegrees
        )
        val targetWidth = targetBitmap.width
        val targetHeight = targetBitmap.height
        val scaleX = targetWidth.toFloat() / orientedBitmap.width
        val scaleY = targetHeight.toFloat() / orientedBitmap.height
        val scale = if (centerCrop) max(scaleX, scaleY) else min(scaleX, scaleY)
        val scaledWidth = (orientedBitmap.width * scale).roundToInt()
        val scaledHeight = (orientedBitmap.height * scale).roundToInt()
        val left = (targetWidth - scaledWidth) / 2
        val top = (targetHeight - scaledHeight) / 2
        val dst = Rect(left, top, left + scaledWidth, top + scaledHeight)

        Canvas(targetBitmap).apply {
            drawColor(Color.BLACK)
            drawBitmap(orientedBitmap, null, dst, filterPaint)
        }
        if (orientedBitmap !== bitmap) {
            orientedBitmap.recycle()
        }
        return targetBitmap
    }

    private fun orientBitmapForCamera(
        bitmap: Bitmap,
        rotateForCamera: Boolean,
        isLandscape: Boolean,
        isFrontCamera: Boolean,
        rotationDegrees: Int?
    ): Bitmap {
        if (!rotateForCamera) return bitmap

        val fallbackDegrees = if (isLandscape) 0 else if (isFrontCamera) 90 else 270
        val degrees = (rotationDegrees ?: fallbackDegrees).floorMod(360)
        if (degrees == 0) return bitmap

        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
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

        for (pixel in pixels) {
            byteBuffer.putFloat((((pixel shr 16) and 0xFF) - inputMean) / inputStd)
            byteBuffer.putFloat((((pixel shr 8) and 0xFF) - inputMean) / inputStd)
            byteBuffer.putFloat(((pixel and 0xFF) - inputMean) / inputStd)
        }
        byteBuffer.rewind()
    }

    /**
     * Process grayscale image for 1-channel classification models
     * Optimized for handwriting recognition (EMNIST-like models)
     * 
     * @param bitmap Input bitmap to process
     * @param targetWidth Target width for the model
     * @param targetHeight Target height for the model  
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
        
        // Allocate ByteBuffer for 1-channel float32 data
        val byteBuffer = ByteBuffer.allocateDirect(targetWidth * targetHeight * 4) // 4 bytes per float
        byteBuffer.order(ByteOrder.nativeOrder())
        
        // Process each pixel
        val pixels = IntArray(targetWidth * targetHeight)
        scaledBitmap.getPixels(pixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)
        
        for (pixel in pixels) {
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
            
            byteBuffer.putFloat(normalizedValue)
        }
        
        // Clean up scaled bitmap if it's different from input
        if (scaledBitmap !== bitmap) {
            scaledBitmap.recycle()
        }
        
        byteBuffer.rewind()
        return byteBuffer
    }


}
