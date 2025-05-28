// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

fun polygonIntersection(subject: List<PointF>, clip: List<PointF>): List<PointF> {
    var outputList = subject.toMutableList()
    val clipEdgeCount = clip.size
    val closedClip = clip + clip.first()
    for (i in 0 until clipEdgeCount) {
        val clipStart = closedClip[i]
        val clipEnd = closedClip[i + 1]
        val inputList = outputList.toList()
        outputList.clear()
        if (inputList.isEmpty()) break
        val closedInput = inputList + inputList.first()
        for (j in 0 until closedInput.size - 1) {
            val current = closedInput[j]
            val next = closedInput[j + 1]
            val currentInside = isInside(current, clipStart, clipEnd)
            val nextInside = isInside(next, clipStart, clipEnd)
            when {
                currentInside && nextInside -> outputList.add(next)
                currentInside && !nextInside -> {
                    computeIntersection(current, next, clipStart, clipEnd)?.let { outputList.add(it) }
                }
                !currentInside && nextInside -> {
                    computeIntersection(current, next, clipStart, clipEnd)?.let { outputList.add(it) }
                    outputList.add(next)
                }
            }
        }
    }
    return outputList
}

fun isInside(point: PointF, edgeStart: PointF, edgeEnd: PointF): Boolean {
    val cross = (edgeEnd.x - edgeStart.x) * (point.y - edgeStart.y) -
            (edgeEnd.y - edgeStart.y) * (point.x - edgeStart.x)
    return cross >= 0
}

fun computeIntersection(p1: PointF, p2: PointF, clipStart: PointF, clipEnd: PointF): PointF? {
    val x1 = p1.x; val y1 = p1.y
    val x2 = p2.x; val y2 = p2.y
    val x3 = clipStart.x; val y3 = clipStart.y
    val x4 = clipEnd.x; val y4 = clipEnd.y
    val denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if (abs(denom) < 1e-10) return null
    val t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    val ix = x1 + t * (x2 - x1)
    val iy = y1 + t * (y2 - y1)
    return PointF(ix, iy)
}

fun polygonArea(poly: List<PointF>): Float {
    if (poly.size < 3) return 0f
    var area = 0f
    for (i in 0 until poly.size - 1) {
        area += poly[i].x * poly[i + 1].y - poly[i + 1].x * poly[i].y
    }
    area += poly.last().x * poly.first().y - poly.first().x * poly.last().y
    return abs(area) * 0.5f
}

fun obbIoU(box1: OBB, box2: OBB): Float {
    val poly1 = box1.toPolygon()
    val poly2 = box2.toPolygon()
    val area1 = box1.area
    val area2 = box2.area
    val interPoly = polygonIntersection(poly1, poly2)
    val interArea = polygonArea(interPoly)
    val unionArea = area1 + area2 - interArea
    return if (unionArea <= 0f) 0f else interArea / unionArea
}

fun nonMaxSuppression(boxes: List<RectF>, scores: List<Float>, iouThreshold: Float): List<Int> {
    val sortedIndices = scores.indices.sortedByDescending { scores[it] }
    val selected = mutableListOf<Int>()
    val active = BooleanArray(boxes.size) { true }
    for (i in sortedIndices.indices) {
        val idx = sortedIndices[i]
        if (!active[idx]) continue
        selected.add(idx)
        for (j in i + 1 until sortedIndices.size) {
            val idxB = sortedIndices[j]
            if (active[idxB]) {
                val iou = computeIoU(boxes[idx], boxes[idxB])
                if (iou > iouThreshold) {
                    active[idxB] = false
                }
            }
        }
    }
    return selected
}

fun computeIoU(a: RectF, b: RectF): Float {
    val interLeft = max(a.left, b.left)
    val interTop = max(a.top, b.top)
    val interRight = min(a.right, b.right)
    val interBottom = min(a.bottom, b.bottom)
    val interArea = max(0f, interRight - interLeft) * max(0f, interBottom - interTop)
    val unionArea = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - interArea
    return if (unionArea > 0) interArea / unionArea else 0f
}

fun nonMaxSuppressionOBB(boxes: List<OBB>, scores: List<Float>, iouThreshold: Float): List<Int> {
    val sortedIndices = scores.indices.sortedByDescending { scores[it] }
    val selected = mutableListOf<Int>()
    val active = BooleanArray(boxes.size) { true }
    for (i in sortedIndices.indices) {
        val idx = sortedIndices[i]
        if (!active[idx]) continue
        selected.add(idx)
        val boxA = boxes[idx]
        for (j in i + 1 until sortedIndices.size) {
            val idxB = sortedIndices[j]
            if (active[idxB]) {
                if (obbIoU(boxA, boxes[idxB]) > iouThreshold) {
                    active[idxB] = false
                }
            }
        }
    }
    return selected
}
