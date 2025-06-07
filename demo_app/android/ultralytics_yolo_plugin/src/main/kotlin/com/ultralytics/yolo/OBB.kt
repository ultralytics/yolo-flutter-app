// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.graphics.PointF

data class OBB(
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val angle: Float   // radians
) {
    /** Area (w × h) */
    val area: Float get() = w * h

    /** Convert this OBB to 4 vertices (list of PointF) */
    fun toPolygon(): List<PointF> {
        val halfW = w / 2
        val halfH = h / 2
        // 4 points in local coordinate system (top-left, top-right, bottom-right, bottom-left)
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
