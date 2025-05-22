package com.ultralytics.yolo

import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import android.graphics.Matrix;

interface Predictor {
    /**
     * Predict method for processing a bitmap
     * @param bitmap Input bitmap to process
     * @param origWidth Original width of the source image
     * @param origHeight Original height of the source image
     * @param rotateForCamera Whether this is a camera feed that requires rotation (true) or a single image (false)
     * @return YOLOResult containing detection results
     */
    fun predict(bitmap: Bitmap, origWidth: Int, origHeight: Int, rotateForCamera: Boolean = false): YOLOResult
    
    abstract fun setIouThreshold(iou: Double)
    abstract fun setConfidenceThreshold(conf: Double)
    abstract fun setNumItemsThreshold(progress: Int)

    var labels: List<String>
    var isUpdating: Boolean
    var inputSize: Size
}

abstract class BasePredictor : Predictor {
    override var isUpdating: Boolean = false
    override lateinit var labels: List<String>
    protected lateinit var interpreter: Interpreter
    override lateinit var inputSize: Size
    protected lateinit var modelInputSize: Pair<Int, Int>
    protected fun isInterpreterInitialized() = this::interpreter.isInitialized

    // 時間計測用（nanosecond 単位など）
    protected var t0: Long = 0L
    protected var t2: Double = 0.0
    protected var t3: Long = System.nanoTime()
    protected var t4: Double = 0.0

    var CONFIDENCE_THRESHOLD:Float = 0.25f
    var IOU_THRESHOLD:Float = 0.25f
    var transformationMatrix: Matrix? = null
    var pendingBitmapFrame: Bitmap? = null

    /** 推論後の時間更新（平滑化） */
    protected fun updateTiming() {
        val now = System.nanoTime()
        val dt = (now - t0) / 1e9
        t2 = 0.05 * dt + 0.95 * t2
        t4 = 0.05 * ((now - t3) / 1e9) + 0.95 * t4
        t3 = now
    }
    override fun setIouThreshold(iou: Double) {

    }

    override fun setConfidenceThreshold(conf: Double) {

    }

    override fun setNumItemsThreshold(progress: Int) {

    }
}
