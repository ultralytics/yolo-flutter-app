package com.ultralytics.ultralytics_yolo;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.YuvImage;
import androidx.camera.core.ImageProxy;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;

public class ImageUtils {
    public static Bitmap toBitmap(ImageProxy imageProxy) {
        byte[] nv21 = yuv420888ToNv21(imageProxy);
        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, imageProxy.getWidth(), imageProxy.getHeight(), null);
        return yuvImageToBitmap(yuvImage);
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
    public static Matrix getTransformationMatrix(
            final int srcWidth,
            final int srcHeight,
            final int dstWidth,
            final int dstHeight,
            final int applyRotation,
            final boolean maintainAspectRatio) {
        final Matrix matrix = new Matrix();

        // Translate so center of image is at origin.
        matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

        // Rotate around origin.
        matrix.postRotate(applyRotation);

        // Account for the already applied rotation, if any, and then determine how
        // much scaling is needed for each axis.
        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

        final int inWidth = transpose ? srcHeight : srcWidth;
        final int inHeight = transpose ? srcWidth : srcHeight;

        // Apply scaling if necessary.
        if (inWidth != dstWidth || inHeight != dstHeight) {
            final float scaleFactorX = dstWidth / (float) inWidth;
            final float scaleFactorY = dstHeight / (float) inHeight;

            if (maintainAspectRatio) {
                // Scale by minimum factor so that dst is filled completely while
                // maintaining the aspect ratio. Some image may fall off the edge.
                final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
                matrix.postScale(scaleFactor, scaleFactor);
            } else {
                // Scale exactly to fill dst from src.
                matrix.postScale(scaleFactorX, scaleFactorY);
            }
        }

        if (applyRotation != 0) {
            // Translate back from origin centered reference to destination frame.
            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
        }

        return matrix;
    }
    private static Bitmap yuvImageToBitmap(YuvImage yuvImage) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        if (!yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out))
            return null;
        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private static byte[] yuv420888ToNv21(ImageProxy imageProxy) {
        int pixelCount = imageProxy.getCropRect().width() * imageProxy.getCropRect().height();
        int pixelSizeBits = ImageFormat.getBitsPerPixel(ImageFormat.YUV_420_888);
        byte[] outputBuffer = new byte[pixelCount * pixelSizeBits / 8];
        imageToByteBuffer(imageProxy, outputBuffer, pixelCount);
        return outputBuffer;
    }

    private static void imageToByteBuffer(ImageProxy imageProxy, byte[] outputBuffer, int pixelCount) {
        assert imageProxy.getFormat() == ImageFormat.YUV_420_888;

        Rect imageCrop = imageProxy.getCropRect();
        ImageProxy.PlaneProxy[] imagePlanes = imageProxy.getPlanes();

        for (int planeIndex = 0; planeIndex < imagePlanes.length; planeIndex++) {
            int outputStride;
            int outputOffset;

            switch (planeIndex) {
                case 0:
                    outputStride = 1;
                    outputOffset = 0;
                    break;
                case 1:
                    outputStride = 2;
                    outputOffset = pixelCount + 1;
                    break;
                case 2:
                    outputStride = 2;
                    outputOffset = pixelCount;
                    break;
                default:
                    return;
            }

            ImageProxy.PlaneProxy plane = imagePlanes[planeIndex];
            ByteBuffer planeBuffer = plane.getBuffer();
            int rowStride = plane.getRowStride();
            int pixelStride = plane.getPixelStride();

            Rect planeCrop = (planeIndex == 0) ?
                    imageCrop :
                    new Rect(imageCrop.left / 2, imageCrop.top / 2, imageCrop.right / 2, imageCrop.bottom / 2);

            int planeWidth = planeCrop.width();
            int planeHeight = planeCrop.height();

            byte[] rowBuffer = new byte[plane.getRowStride()];

            int rowLength = (pixelStride == 1 && outputStride == 1) ? planeWidth : (planeWidth - 1) * pixelStride + 1;

            for (int row = 0; row < planeHeight; row++) {
                planeBuffer.position((row + planeCrop.top) * rowStride + planeCrop.left * pixelStride);

                if (pixelStride == 1 && outputStride == 1) {
                    planeBuffer.get(outputBuffer, outputOffset, rowLength);
                    outputOffset += rowLength;
                } else {
                    planeBuffer.get(rowBuffer, 0, rowLength);
                    for (int col = 0; col < planeWidth; col++) {
                        outputBuffer[outputOffset] = rowBuffer[col * pixelStride];
                        outputOffset += outputStride;
                    }
                }
            }
        }
    }
}
