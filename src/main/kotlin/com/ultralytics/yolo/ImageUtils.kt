// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.graphics.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.max

object ImageUtils {

    /**
     * Returns a transformation matrix from one reference frame into another. Handles cropping (if
     * maintaining aspect ratio is desired) and rotation.
     *
     * @param srcWidth            Width of source frame.
     * @param srcHeight           Height of source frame.
     * @param dstWidth            Width of destination frame.
     * @param dstHeight           Height of destination frame.
     * @param applyRotation       Amount of rotation to apply from one frame to another. Must be a multiple
     *                            of 90.
     * @param maintainAspectRatio If true, will ensure that scaling in x and y remains constant,
     *                            cropping the image if necessary.
     * @return The transformation fulfilling the desired requirements.
     */
    @JvmStatic
    fun getTransformationMatrix(
        srcWidth: Int,
        srcHeight: Int,
        dstWidth: Int,
        dstHeight: Int,
        applyRotation: Int,
        maintainAspectRatio: Boolean
    ): Matrix {
        val matrix = Matrix()

        // Translate so center of image is at origin.
        matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f)

        // Rotate around origin.
        matrix.postRotate(applyRotation.toFloat())

        // Check if we need to swap width/height (e.g., for 90 or 270 degrees rotation)
        val transpose = (abs(applyRotation) + 90) % 180 == 0

        val inWidth = if (transpose) srcHeight else srcWidth
        val inHeight = if (transpose) srcWidth else srcHeight

        // Apply scaling if necessary.
        if (inWidth != dstWidth || inHeight != dstHeight) {
            val scaleFactorX = dstWidth.toFloat() / inWidth
            val scaleFactorY = dstHeight.toFloat() / inHeight

            if (maintainAspectRatio) {
                // Scale by min factor so that dst is filled completely while
                // maintaining the aspect ratio (some image may fall off the edge).
                val scaleFactor = max(scaleFactorX, scaleFactorY)
                matrix.postScale(scaleFactor, scaleFactor)
            } else {
                // Scale exactly to fill dst from src.
                matrix.postScale(scaleFactorX, scaleFactorY)
            }
        }

        if (applyRotation != 0) {
            // Translate back from origin-centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f)
        }

        return matrix
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
        // Scale bitmap to target size
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        
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
        if (scaledBitmap != bitmap) {
            scaledBitmap.recycle()
        }
        
        byteBuffer.rewind()
        return byteBuffer
    }
}
