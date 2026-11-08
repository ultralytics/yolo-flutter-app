// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import QuartzCore
import UIKit
import UltralyticsYOLO

// Annotation colors, mirroring ultralytics/utils/plotting.py Colors. Opaque; callers apply alpha where they draw.
let ultralyticsColors: [UIColor] = [
  UIColor(red: 4 / 255, green: 42 / 255, blue: 255 / 255, alpha: 1),
  UIColor(red: 11 / 255, green: 219 / 255, blue: 235 / 255, alpha: 1),
  UIColor(red: 243 / 255, green: 243 / 255, blue: 243 / 255, alpha: 1),
  UIColor(red: 0 / 255, green: 223 / 255, blue: 183 / 255, alpha: 1),
  UIColor(red: 17 / 255, green: 31 / 255, blue: 104 / 255, alpha: 1),
  UIColor(red: 255 / 255, green: 111 / 255, blue: 221 / 255, alpha: 1),
  UIColor(red: 255 / 255, green: 68 / 255, blue: 79 / 255, alpha: 1),
  UIColor(red: 204 / 255, green: 237 / 255, blue: 0 / 255, alpha: 1),
  UIColor(red: 0 / 255, green: 243 / 255, blue: 68 / 255, alpha: 1),
  UIColor(red: 189 / 255, green: 0 / 255, blue: 255 / 255, alpha: 1),
  UIColor(red: 0 / 255, green: 180 / 255, blue: 255 / 255, alpha: 1),
  UIColor(red: 221 / 255, green: 0 / 255, blue: 186 / 255, alpha: 1),
  UIColor(red: 0 / 255, green: 255 / 255, blue: 255 / 255, alpha: 1),
  UIColor(red: 38 / 255, green: 192 / 255, blue: 0 / 255, alpha: 1),
  UIColor(red: 1 / 255, green: 255 / 255, blue: 179 / 255, alpha: 1),
  UIColor(red: 125 / 255, green: 36 / 255, blue: 255 / 255, alpha: 1),
  UIColor(red: 123 / 255, green: 0 / 255, blue: 104 / 255, alpha: 1),
  UIColor(red: 255 / 255, green: 27 / 255, blue: 108 / 255, alpha: 1),
  UIColor(red: 252 / 255, green: 109 / 255, blue: 47 / 255, alpha: 1),
  UIColor(red: 162 / 255, green: 255 / 255, blue: 11 / 255, alpha: 1),
]

private let posePalette: [UIColor] =
  ([
    [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
    [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
    [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
    [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
    [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255],
  ] as [[CGFloat]]).map {
    UIColor(red: $0[0] / 255, green: $0[1] / 255, blue: $0[2] / 255, alpha: 1)
  }

private let limbColorIndices = [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]
private let kptColorIndices = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]

private let skeleton = [
  [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
  [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
  [1, 3], [2, 4], [3, 5], [4, 6],
]

private let darkLabelTextColor = UIColor(red: 17 / 255, green: 31 / 255, blue: 104 / 255, alpha: 1)

/// Label text color on a class-colored background: #111F68 when its luma exceeds 150, else white.
func labelTextColor(for background: UIColor) -> UIColor {
  var red: CGFloat = 0
  var green: CGFloat = 0
  var blue: CGFloat = 0
  background.getRed(&red, green: &green, blue: &blue, alpha: nil)
  return (0.299 * red + 0.587 * green + 0.114 * blue) * 255 > 150 ? darkLabelTextColor : .white
}

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
  for (i, keypoints) in keypointsList.enumerated()
  where i < confsList.count && i < boundingBoxes.count {
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
  // COCO-17 layouts use the pose palette and skeleton; other layouts color keypoint k with the class palette.
  let coco = numKeypoints == kptColorIndices.count
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
      let color =
        coco ? posePalette[kptColorIndices[i]] : ultralyticsColors[i % ultralyticsColors.count]
      drawCircle(on: layer, at: point, radius: radius, color: color)
    }
  }

  guard drawSkeleton, coco else { return }
  for (index, bone) in skeleton.enumerated() {
    let start = points[bone[0]]
    let end = points[bone[1]]
    if start.1 >= confThreshold && end.1 >= confThreshold {
      drawLine(
        on: layer, from: start.0, to: end.0, color: posePalette[limbColorIndices[index]],
        lineWidth: lineWidth)
    }
  }
}

private func drawCircle(on layer: CALayer, at point: CGPoint, radius: CGFloat, color: UIColor) {
  let circleLayer = CAShapeLayer()
  circleLayer.path =
    UIBezierPath(
      arcCenter: point, radius: radius, startAngle: 0, endAngle: .pi * 2, clockwise: true
    ).cgPath
  circleLayer.fillColor = color.cgColor
  layer.addSublayer(circleLayer)
}

private func drawLine(
  on layer: CALayer,
  from start: CGPoint,
  to end: CGPoint,
  color: UIColor,
  lineWidth: CGFloat = 2
) {
  let lineLayer = CAShapeLayer()
  let path = UIBezierPath()
  path.move(to: start)
  path.addLine(to: end)
  lineLayer.path = path.cgPath
  lineLayer.lineWidth = max(lineWidth, 1.5)
  lineLayer.strokeColor = color.cgColor
  layer.addSublayer(lineLayer)
}
