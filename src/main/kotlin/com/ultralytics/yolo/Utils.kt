package com.ultralytics.yolo

import android.graphics.Color
import android.util.Log
import java.io.File
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * YOLO 工具类
 * 提供模型加载、颜色定义等通用功能
 */
object YOLOUtils {
    private const val TAG = "YOLOUtils"
    
    /**
     * Ultralytics 官方配色方案 (透明度 153/255)
     */
    val ultralyticsColors: List<Int> = listOf(
        Color.argb(153, 4, 42, 255),
        Color.argb(153, 11, 219, 235),
        Color.argb(153, 243, 243, 243),
        Color.argb(153, 0, 223, 183),
        Color.argb(153, 17, 31, 104),
        Color.argb(153, 255, 111, 221),
        Color.argb(153, 255, 68, 79),
        Color.argb(153, 204, 237, 0),
        Color.argb(153, 0, 243, 68),
        Color.argb(153, 189, 0, 255),
        Color.argb(153, 0, 180, 255),
        Color.argb(153, 221, 0, 186),
        Color.argb(153, 0, 255, 255),
        Color.argb(153, 38, 192, 0),
        Color.argb(153, 1, 255, 179),
        Color.argb(153, 125, 36, 255),
        Color.argb(153, 123, 0, 104),
        Color.argb(153, 255, 27, 108),
        Color.argb(153, 252, 109, 47),
        Color.argb(153, 162, 255, 11)
    )

    /**
     * 确保模型路径包含 .tflite 后缀
     * @param modelPath 模型路径
     * @return 包含后缀的路径
     */
    fun ensureTFLiteExtension(modelPath: String): String {
        return if (!modelPath.lowercase().endsWith(".tflite")) {
            "$modelPath.tflite"
        } else {
            modelPath
        }
    }

    /**
     * 加载模型文件 (仅支持绝对路径)
     * 
     * @param modelPath 模型的本地绝对路径
     * @return 映射的字节缓冲区
     * @throws RuntimeException 如果模型文件不存在或加载失败
     */
    fun loadModelFile(modelPath: String): MappedByteBuffer {
        val finalPath = ensureTFLiteExtension(modelPath)
        val file = File(finalPath)
        
        if (file.exists() && file.isFile) {
            Log.d(TAG, "从文件系统加载模型: $finalPath")
            try {
                val channel = java.io.RandomAccessFile(file, "r").channel
                return channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length())
            } catch (e: Exception) {
                Log.e(TAG, "读取模型文件出错: $finalPath", e)
                throw RuntimeException("读取模型文件失败: ${e.message}")
            }
        } else {
            val errorMsg = "模型文件未找到: $finalPath (请确保使用了正确的绝对路径)"
            Log.e(TAG, errorMsg)
            throw RuntimeException(errorMsg)
        }
    }
}
