// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import QuartzCore
import UIKit
import YOLO

let ultralyticsColors: [UIColor] = [
  UIColor(red: 4 / 255, green: 42 / 255, blue: 255 / 255, alpha: 0.6),
  UIColor(red: 11 / 255, green: 219 / 255, blue: 235 / 255, alpha: 0.6),
  UIColor(red: 243 / 255, green: 243 / 255, blue: 243 / 255, alpha: 0.6),
  UIColor(red: 0 / 255, green: 223 / 255, blue: 183 / 255, alpha: 0.6),
  UIColor(red: 17 / 255, green: 31 / 255, blue: 104 / 255, alpha: 0.6),
  UIColor(red: 255 / 255, green: 111 / 255, blue: 221 / 255, alpha: 0.6),
  UIColor(red: 255 / 255, green: 68 / 255, blue: 79 / 255, alpha: 0.6),
  UIColor(red: 204 / 255, green: 237 / 255, blue: 0 / 255, alpha: 0.6),
  UIColor(red: 0 / 255, green: 243 / 255, blue: 68 / 255, alpha: 0.6),
  UIColor(red: 189 / 255, green: 0 / 255, blue: 255 / 255, alpha: 0.6),
  UIColor(red: 0 / 255, green: 180 / 255, blue: 255 / 255, alpha: 0.6),
  UIColor(red: 221 / 255, green: 0 / 255, blue: 186 / 255, alpha: 0.6),
  UIColor(red: 0 / 255, green: 255 / 255, blue: 255 / 255, alpha: 0.6),
  UIColor(red: 38 / 255, green: 192 / 255, blue: 0 / 255, alpha: 0.6),
  UIColor(red: 1 / 255, green: 255 / 255, blue: 179 / 255, alpha: 0.6),
  UIColor(red: 125 / 255, green: 36 / 255, blue: 255 / 255, alpha: 0.6),
  UIColor(red: 123 / 255, green: 0 / 255, blue: 104 / 255, alpha: 0.6),
  UIColor(red: 255 / 255, green: 27 / 255, blue: 108 / 255, alpha: 0.6),
  UIColor(red: 252 / 255, green: 109 / 255, blue: 47 / 255, alpha: 0.6),
  UIColor(red: 162 / 255, green: 255 / 255, blue: 11 / 255, alpha: 0.6),
]

private let posePalette: [[CGFloat]] = [
  [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
  [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
  [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
  [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
  [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255],
]

private let limbColorIndices = [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]
private let kptColorIndices = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]

private let skeleton = [
  [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
  [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
  [2, 4], [3, 5], [4, 6], [5, 7],
]

func drawKeypoints(
  keypointsList: [[(x: Float, y: Float)]],
  confsList: [[Float]],
  boundingBoxes: [Box],
  on layer: CALayer,
  imageViewSize: CGSize,
  originalImageSize: CGSize,
  radius: CGFloat = 5,
  confThreshold: Float = 0.25,
  drawSkeleton: Bool = true
) {
  let scaledRadius = max(imageViewSize.width, imageViewSize.height) / 100
  for (i, keypoints) in keypointsList.enumerated() where i < confsList.count && i < boundingBoxes.count {
    drawSinglePersonKeypoints(
      keypoints: keypoints, confs: confsList[i], boundingBox: boundingBoxes[i],
      on: layer,
      imageViewSize: imageViewSize,
      radius: scaledRadius,
      confThreshold: confThreshold,
      drawSkeleton: drawSkeleton
    )
  }
}

private func drawSinglePersonKeypoints(
  keypoints: [(x: Float, y: Float)],
  confs: [Float],
  boundingBox: Box,
  on layer: CALayer,
  imageViewSize: CGSize,
  radius: CGFloat,
  confThreshold: Float,
  drawSkeleton: Bool
) {
  let lineWidth = radius * 0.4
  let numKeypoints = keypoints.count
  var points: [(CGPoint, Float)] = Array(repeating: (CGPoint.zero, 0), count: numKeypoints)

  for i in 0..<numKeypoints where i < confs.count {
    let point = CGPoint(
      x: CGFloat(keypoints[i].x) * imageViewSize.width,
      y: CGFloat(keypoints[i].y) * imageViewSize.height)
    let confidence = confs[i]
    if confidence >= confThreshold
      && boundingBox.xywhn.contains(CGPoint(x: CGFloat(keypoints[i].x), y: CGFloat(keypoints[i].y)))
    {
      points[i] = (point, confidence)
      let colorIndex = i < kptColorIndices.count ? kptColorIndices[i] : i % posePalette.count
      drawCircle(on: layer, at: point, radius: radius, color: colorIndex)
    }
  }

  guard drawSkeleton, numKeypoints == 17 else { return }
  for (index, bone) in skeleton.enumerated() {
    let (startIdx, endIdx) = (bone[0] - 1, bone[1] - 1)
    guard startIdx < points.count, endIdx < points.count else { continue }
    let start = points[startIdx]
    let end = points[endIdx]
    if start.1 >= confThreshold && end.1 >= confThreshold {
      let colorIndex =
        index < limbColorIndices.count ? limbColorIndices[index] : index % posePalette.count
      drawLine(on: layer, from: start.0, to: end.0, color: colorIndex, lineWidth: lineWidth)
    }
  }
}

private func drawCircle(on layer: CALayer, at point: CGPoint, radius: CGFloat, color index: Int) {
  let circleLayer = CAShapeLayer()
  circleLayer.path = UIBezierPath(
    arcCenter: point, radius: radius, startAngle: 0, endAngle: .pi * 2, clockwise: true
  ).cgPath
  let color = posePalette[index].map { $0 / 255.0 }
  circleLayer.fillColor =
    UIColor(red: color[0], green: color[1], blue: color[2], alpha: 1.0).cgColor
  layer.addSublayer(circleLayer)
}

private func drawLine(
  on layer: CALayer,
  from start: CGPoint,
  to end: CGPoint,
  color index: Int,
  lineWidth: CGFloat = 2
) {
  let lineLayer = CAShapeLayer()
  let path = UIBezierPath()
  path.move(to: start)
  path.addLine(to: end)
  lineLayer.path = path.cgPath
  lineLayer.lineWidth = max(lineWidth, 1.5)
  let color = posePalette[index].map { $0 / 255.0 }
  lineLayer.strokeColor =
    UIColor(red: color[0], green: color[1], blue: color[2], alpha: 1.0).cgColor
  layer.addSublayer(lineLayer)
}
