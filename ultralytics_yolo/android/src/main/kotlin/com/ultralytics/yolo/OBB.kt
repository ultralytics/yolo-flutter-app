package com.ultralytics.yolo

import android.graphics.PointF

data class OBB(
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val angle: Float   // ラジアン
) {
    /** 面積 (w × h) */
    val area: Float get() = w * h

    /** この OBB を4頂点（PointF のリスト）に変換 */
    fun toPolygon(): List<PointF> {
        val halfW = w / 2
        val halfH = h / 2
        // ローカル座標系での4点（左上、右上、右下、左下）
        val localCorners = listOf(
            PointF(-halfW, -halfH),
            PointF(halfW, -halfH),
            PointF(halfW, halfH),
            PointF(-halfW, halfH)
        )
        val cosA = kotlin.math.cos(angle)
        val sinA = kotlin.math.sin(angle)
        return localCorners.map { pt ->
            PointF(
                (cosA * pt.x - sinA * pt.y) + cx,
                (sinA * pt.x + cosA * pt.y) + cy
            )
        }
    }
}
