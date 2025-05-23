// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.graphics.*
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import kotlin.math.abs
import kotlin.math.max

object ImageUtils {

    /**
     * ImageProxy を NV21(BYTE配列) に変換し、[YuvImage] -> [Bitmap] へ変換するサンプル
     */
    @JvmStatic
    fun toBitmap(imageProxy: ImageProxy): Bitmap? {
        val nv21 = yuv420888ToNv21(imageProxy)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        return yuvImageToBitmap(yuvImage)
    }

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
     * [YuvImage] を [Bitmap] に変換するサンプル
     */
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

    /**
     * [ImageProxy] を NV21 (byte配列) に変換する。
     */
    private fun yuv420888ToNv21(imageProxy: ImageProxy): ByteArray {
        val cropRect = imageProxy.cropRect
        val pixelCount = cropRect.width() * cropRect.height()
        val pixelSizeBits = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888)
        val outputBuffer = ByteArray(pixelCount * pixelSizeBits / 8)
        imageToByteBuffer(imageProxy, outputBuffer, pixelCount)
        return outputBuffer
    }

    /**
     * [ImageProxy] の各 Plane(Y/U/V) から画素を読み取って、NV21 バイト列 ([outputBuffer]) に書き込む。
     */
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

            // Y plane は cropRect をそのまま利用
            // U / V plane は cropRect を 1/2 に縮小
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

            // 1ピクセルずつ読み出してもよいが、pixelStrideやoutputStrideが1の場合はまとめて読める
            val rowLength = if (pixelStride == 1 && outputStride == 1) {
                planeWidth
            } else {
                (planeWidth - 1) * pixelStride + 1
            }

            for (row in 0 until planeHeight) {
                // 現在の行の先頭位置まで ByteBuffer を進める
                planeBuffer.position(
                    (row + planeCrop.top) * rowStride +
                            planeCrop.left * pixelStride
                )

                if (pixelStride == 1 && outputStride == 1) {
                    // まとめてコピーできる場合
                    planeBuffer.get(outputBuffer, outputOffset, rowLength)
                    outputOffset += rowLength
                } else {
                    // 1ピクセルごとにコピーが必要
                    planeBuffer.get(rowBuffer, 0, rowLength)
                    for (col in 0 until planeWidth) {
                        outputBuffer[outputOffset] = rowBuffer[col * pixelStride]
                        outputOffset += outputStride
                    }
                }
            }
        }
    }
}
