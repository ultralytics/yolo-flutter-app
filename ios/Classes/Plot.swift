// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Ultralytics YOLO Package, providing visualization utilities.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The Plot module provides visualization utilities for rendering YOLO model results.
//  It includes functions for drawing bounding boxes, segmentation masks, pose keypoints,
//  classification results, and oriented bounding boxes on images. The module implements
//  specialized rendering algorithms for each type of prediction, handles color management
//  for different classes, and supports both static image and real-time visualization scenarios.
//  Each visualization function is optimized for the specific task to provide clear and
//  informative visual feedback to users with minimal performance impact.

import Accelerate
import CoreImage
import CoreML
import Foundation
import QuartzCore
import UIKit

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

let posePalette: [[CGFloat]] = [
  [255, 128, 0],
  [255, 153, 51],
  [255, 178, 102],
  [230, 230, 0],
  [255, 153, 255],
  [153, 204, 255],
  [255, 102, 255],
  [255, 51, 255],
  [102, 178, 255],
  [51, 153, 255],
  [255, 153, 153],
  [255, 102, 102],
  [255, 51, 51],
  [153, 255, 153],
  [102, 255, 102],
  [51, 255, 51],
  [0, 255, 0],
  [0, 0, 255],
  [255, 0, 0],
  [255, 255, 255],
]

let limbColorIndices = [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]
let kptColorIndices = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]

let skeleton = [
  [16, 14],
  [14, 12],
  [17, 15],
  [15, 13],
  [12, 13],
  [6, 12],
  [7, 13],
  [6, 7],
  [6, 8],
  [7, 9],
  [8, 10],
  [9, 11],
  [2, 3],
  [1, 2],
  [1, 3],
  [2, 4],
  [3, 5],
  [4, 6],
  [5, 7],
]

/// Calculate smart label position that ensures the label stays within screen bounds.
///
/// - Parameters:
///   - boxRect: The bounding box rectangle.
///   - labelSize: The size of the label.
///   - screenSize: The size of the screen/image.
/// - Returns: The adjusted rectangle for the label.
func calculateSmartLabelRect(boxRect: CGRect, labelSize: CGSize, screenSize: CGSize) -> CGRect {
  // Initial position: above the box
  var labelX = boxRect.minX
  var labelY = boxRect.minY - labelSize.height

  // Check top boundary
  if labelY < 0 {
    // Place inside top of box
    labelY = boxRect.minY
  }

  // Check left boundary
  if labelX < 0 {
    labelX = 0
  }

  // Check right boundary
  if labelX + labelSize.width > screenSize.width {
    labelX = screenSize.width - labelSize.width
    // If still too wide, align with box's right edge
    if labelX < 0 {
      labelX = max(0, boxRect.maxX - labelSize.width)
    }
  }

  // Check bottom boundary
  if labelY + labelSize.height > screenSize.height {
    labelY = screenSize.height - labelSize.height
  }

  return CGRect(x: labelX, y: labelY, width: labelSize.width, height: labelSize.height)
}

/// Executes `body` inside a bitmap graphics context rendered at pixel scale.
///
/// Flips the y-axis so drawing matches UIKit's top-left origin and draws `cgImage` as the
/// background. The same boilerplate appeared across every `draw…` visualization helper.
private func renderWithBackground(
  _ ciImage: CIImage,
  targetSize: CGSize? = nil,
  _ body: (CGContext, CGSize) -> Void
) -> UIImage? {
  let context = CIContext(options: nil)
  let extent = ciImage.extent
  guard let cgImage = context.createCGImage(ciImage, from: extent) else {
    return nil
  }
  let size = targetSize ?? CGSize(width: cgImage.width, height: cgImage.height)
  UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
  defer { UIGraphicsEndImageContext() }
  guard let drawContext = UIGraphicsGetCurrentContext() else { return nil }

  drawContext.saveGState()
  drawContext.translateBy(x: 0, y: size.height)
  drawContext.scaleBy(x: 1, y: -1)
  drawContext.draw(cgImage, in: CGRect(origin: .zero, size: size))
  drawContext.restoreGState()

  body(drawContext, size)
  return UIGraphicsGetImageFromCurrentImageContext()
}

/// Draws a filled rounded-rect label background with centered text using `DetectionLabelStyle`.
///
/// When `imageSize` is provided the label rect is clamped to `[0, 0, imageSize.width, imageSize.height]` so boxes near the
/// top/left edge don't end up with the badge cropped off the canvas (the upstream `DetectionLabelStyle.frame` always places
/// the badge above/left of the anchor and is unaware of image bounds). Live overlays pass `nil` and rely on the
/// platform-view's own clipping.
private func drawDetectionLabel(
  _ labelText: String,
  in ctx: CGContext,
  fontSize: CGFloat,
  color: UIColor,
  alpha: CGFloat,
  anchor: CGPoint,
  cornerRadius: CGFloat,
  imageSize: CGSize? = nil
) {
  var labelRect = DetectionLabelStyle.frame(for: labelText, fontSize: fontSize, anchor: anchor)
  if let imageSize {
    labelRect = clampLabelRect(labelRect, in: imageSize, anchor: anchor)
  }
  ctx.setFillColor(color.withAlphaComponent(alpha).cgColor)
  let labelPath = UIBezierPath(
    roundedRect: labelRect,
    cornerRadius: min(DetectionLabelStyle.cornerRadius, cornerRadius)
  )
  ctx.addPath(labelPath.cgPath)
  ctx.fillPath()

  let textSize = labelText.size(withAttributes: DetectionLabelStyle.attributes(fontSize: fontSize))
  let textPoint = CGPoint(
    x: labelRect.origin.x + DetectionLabelStyle.horizontalPadding / 2,
    y: labelRect.origin.y + (labelRect.height - textSize.height) / 2
  )
  labelText.draw(
    at: textPoint,
    withAttributes: DetectionLabelStyle.attributes(fontSize: fontSize, alpha: alpha)
  )
}

/// Keeps `labelRect` inside the image: if it would clip off the top it flips below `anchor`, if it would clip off the
/// right/bottom it slides back in, and if it would clip off the left it left-aligns to 0.
private func clampLabelRect(_ rect: CGRect, in imageSize: CGSize, anchor: CGPoint) -> CGRect {
  var origin = rect.origin
  if origin.y < 0 { origin.y = anchor.y }
  if origin.x < 0 { origin.x = 0 }
  if origin.x + rect.width > imageSize.width { origin.x = max(0, imageSize.width - rect.width) }
  if origin.y + rect.height > imageSize.height { origin.y = max(0, imageSize.height - rect.height) }
  return CGRect(origin: origin, size: rect.size)
}

/// Stroked label + box drawing shared by the detection/pose/segmentation renderers.
///
/// - Parameter rounded: pass `true` for the "rounded corner" style used by pose/segment
///   overlays, and `false` for the straight-corner style used by raw detections.
private func drawBoxLabel(
  _ box: Box,
  in ctx: CGContext,
  imageSize: CGSize,
  rounded: Bool
) {
  let color = ultralyticsColors[box.index % ultralyticsColors.count]
  ctx.setStrokeColor(color.cgColor)
  let lineWidth = max(imageSize.width, imageSize.height) / 200
  ctx.setLineWidth(lineWidth)

  let rect = box.xywh
  if rounded {
    let cornerRadius = max(min(rect.width, rect.height) * 0.05, 2.0)
    let boxPath = UIBezierPath(roundedRect: rect, cornerRadius: cornerRadius)
    ctx.addPath(boxPath.cgPath)
    ctx.strokePath()
  } else {
    ctx.stroke(rect)
  }

  let fontSize = max(imageSize.width, imageSize.height) / 50
  let labelText = DetectionLabelStyle.text(className: box.cls, confidence: CGFloat(box.conf))
  drawDetectionLabel(
    labelText,
    in: ctx,
    fontSize: fontSize,
    color: color,
    alpha: 1,
    anchor: rect.origin,
    cornerRadius: max(min(rect.width, rect.height) * 0.05, 2.0),
    imageSize: imageSize
  )
}

public func drawYOLODetections(on ciImage: CIImage, result: YOLOResult) -> UIImage {
  renderWithBackground(ciImage) { ctx, size in
    for box in result.boxes {
      drawBoxLabel(box, in: ctx, imageSize: size, rounded: false)
    }
  } ?? UIImage()
}

private func maskContentRect(
  maskWidth: Int,
  maskHeight: Int,
  inputWidth: Int,
  inputHeight: Int,
  originalImageSize: CGSize
) -> CGRect? {
  // Remove prototype-space letterbox padding before masks are scaled to the original image.
  let modelWidth = CGFloat(inputWidth)
  let modelHeight = CGFloat(inputHeight)
  let originalWidth = originalImageSize.width
  let originalHeight = originalImageSize.height
  guard
    maskWidth > 0,
    maskHeight > 0,
    modelWidth > 0,
    modelHeight > 0,
    originalWidth > 0,
    originalHeight > 0
  else {
    return nil
  }

  let gain = min(modelWidth / originalWidth, modelHeight / originalHeight)
  guard gain > 0 else { return nil }
  let resizedWidth = (originalWidth * gain).rounded()
  let resizedHeight = (originalHeight * gain).rounded()
  // Match Ultralytics LetterBox leading-pad rounding: round(d - 0.1).
  let padX = ((modelWidth - resizedWidth) / 2 - 0.1).rounded()
  let padY = ((modelHeight - resizedHeight) / 2 - 0.1).rounded()
  let scaleX = CGFloat(maskWidth) / modelWidth
  let scaleY = CGFloat(maskHeight) / modelHeight
  let left = min(max(Int((padX * scaleX).rounded()), 0), maskWidth - 1)
  let top = min(max(Int((padY * scaleY).rounded()), 0), maskHeight - 1)
  let right = min(max(Int(((padX + resizedWidth) * scaleX).rounded()), left + 1), maskWidth)
  let bottom = min(max(Int(((padY + resizedHeight) * scaleY).rounded()), top + 1), maskHeight)
  return CGRect(
    x: CGFloat(left),
    y: CGFloat(top),
    width: CGFloat(right - left),
    height: CGFloat(bottom - top))
}

func generateCombinedMaskImage(
  detectedObjects: [(CGRect, Int, Float, MLMultiArray)],
  protos: MLMultiArray,  // shape: [1, C, H, W]
  inputWidth: Int,
  inputHeight: Int,
  threshold: Float = 0.5,
  returnIndividualMasks: Bool = true,
  originalImageSize: CGSize? = nil
) -> (CGImage?, [[[Float]]]?)? {
  // 1) Check protos shape
  let maskHeight = protos.shape[2].intValue  // example: 160
  let maskWidth = protos.shape[3].intValue  // example: 160
  let maskChannels = protos.shape[1].intValue  // example: 32
  guard
    protos.shape.count == 4,
    protos.shape[0].intValue == 1,
    maskHeight > 0,
    maskWidth > 0,
    maskChannels > 0
  else {
    return nil
  }

  let protosPointer = protos.dataPointer.assumingMemoryBound(to: Float.self)
  let HW = maskHeight * maskWidth
  let N = detectedObjects.count

  // 2) Prepare matrix A: (N, C) at once (number of objects x mask channels)
  var coeffsArray = [Float](repeating: 0, count: N * maskChannels)
  for i in 0..<N {
    let (_, _, _, coeffsMLArray) = detectedObjects[i]
    let coeffsPtr = coeffsMLArray.dataPointer.assumingMemoryBound(to: Float.self)
    // Row i of matrix A: write to coeffsArray[i*C .. i*C + C-1]
    for c in 0..<maskChannels {
      coeffsArray[i * maskChannels + c] = coeffsPtr[c]
    }
  }

  // 3) Matrix B: (C, HW) uses protosPointer directly
  //    Memory layout is [1, C, H, W] => (C, H, W) => (C, HW). Rows: C, Columns: HW
  //    vDSP_mmul simply treats contiguous memory as 2D, so this is OK.

  // 4) Matrix C (output): (N, HW) allocate => combinedMask
  //    A flat 1D array with N*HW elements
  var combinedMask = [Float](repeating: 0, count: N * HW)

  // 5) Batch computation with vDSP_mmul: (N x C) * (C x HW) => (N x HW)
  coeffsArray.withUnsafeBufferPointer { Abuf in
    combinedMask.withUnsafeMutableBufferPointer { Cbuf in
      vDSP_mmul(
        Abuf.baseAddress!, 1,  // A
        protosPointer, 1,  // B
        Cbuf.baseAddress!, 1,  // C
        vDSP_Length(N),
        vDSP_Length(HW),
        vDSP_Length(maskChannels)
      )
    }
  }

  // 6) Sort by score (to control drawing order during composition)
  //    => (originalIndex, box, classID, score)
  let indexedObjects: [(Int, CGRect, Int, Float)] =
    detectedObjects.enumerated().map { (i, obj) in (i, obj.0, obj.1, obj.2) }
  let sortedObjects = indexedObjects.sorted { $0.3 < $1.3 }  // ascending by score

  // 7) RGBA buffer (160x160)
  var mergedPixels = [UInt8](repeating: 0, count: HW * 4)
  let scaleX = Float(maskWidth) / Float(inputWidth)
  let scaleY = Float(maskHeight) / Float(inputHeight)

  // 8) Whether to keep individual probability maps
  var probabilityMasks: [[[Float]]]? = nil
  if returnIndividualMasks {
    probabilityMasks = Array(
      repeating: Array(
        repeating: [Float](repeating: 0.0, count: maskWidth),
        count: maskHeight
      ),
      count: N
    )
  }

  // 9) Compose according to sort order
  for (originalIndex, box, classID, _) in sortedObjects {
    // Convert boundingBox to mask coordinate system
    let minX = Int(Float(box.minX) * scaleX)
    let minY = Int(Float(box.minY) * scaleY)
    let maxX = Int(Float(box.maxX) * scaleX)
    let maxY = Int(Float(box.maxY) * scaleY)

    let boxX1 = max(0, min(minX, maskWidth - 1))
    let boxX2 = max(0, min(maxX, maskWidth - 1))
    let boxY1 = max(0, min(minY, maskHeight - 1))
    let boxY2 = max(0, min(maxY, maskHeight - 1))

    let startIdx = originalIndex * HW

    // Get class color
    let _colorIndex = classID % ultralyticsColors.count
    guard let color = ultralyticsColors[_colorIndex].toRGBComponents() else {
      continue
    }
    let r = UInt8(color.red)
    let g = UInt8(color.green)
    let b = UInt8(color.blue)

    // Pixel loop: box range only
    for y in boxY1...boxY2 {
      for x in boxX1...boxX2 {
        let px = y * maskWidth + x
        let maskVal = combinedMask[startIdx + px]
        if maskVal > threshold {
          let pixIndex = px * 4
          mergedPixels[pixIndex + 0] = r
          mergedPixels[pixIndex + 1] = g
          mergedPixels[pixIndex + 2] = b
          mergedPixels[pixIndex + 3] = 255
        }
      }
    }
  }

  if returnIndividualMasks, var masksArray = probabilityMasks {
    for i in 0..<N {
      let startIdx = i * HW
      for k in 0..<HW {
        let row = k / maskWidth
        let col = k % maskWidth
        masksArray[i][row][col] = combinedMask[startIdx + k]
      }
    }
    probabilityMasks = masksArray
  }

  // 11) RGBA buffer -> CGImage
  let colorSpace = CGColorSpaceCreateDeviceRGB()
  let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
  let totalBytes = mergedPixels.count

  guard let providerRef = CGDataProvider(data: NSData(bytes: &mergedPixels, length: totalBytes))
  else {
    return nil
  }
  guard
    let mergedCGImage = CGImage(
      width: maskWidth,
      height: maskHeight,
      bitsPerComponent: 8,
      bitsPerPixel: 32,
      bytesPerRow: maskWidth * 4,
      space: colorSpace,
      bitmapInfo: bitmapInfo,
      provider: providerRef,
      decode: nil,
      shouldInterpolate: false,
      intent: .defaultIntent
    )
  else {
    return nil
  }

  var outputImage = mergedCGImage
  var outputProbabilityMasks = probabilityMasks
  if let originalImageSize,
    let rect = maskContentRect(
      maskWidth: maskWidth,
      maskHeight: maskHeight,
      inputWidth: inputWidth,
      inputHeight: inputHeight,
      originalImageSize: originalImageSize),
    rect != CGRect(x: 0, y: 0, width: CGFloat(maskWidth), height: CGFloat(maskHeight)),
    let croppedImage = mergedCGImage.cropping(to: rect)
  {
    outputImage = croppedImage
    if let masksArray = outputProbabilityMasks {
      let left = Int(rect.minX)
      let top = Int(rect.minY)
      let right = Int(rect.maxX)
      let bottom = Int(rect.maxY)
      outputProbabilityMasks = masksArray.map { mask in
        (top..<bottom).map { row in
          Array(mask[row][left..<right])
        }
      }
    }
  }

  return (outputImage, outputProbabilityMasks)
}

func composeImageWithMask(
  baseImage: CGImage,
  maskImage: CGImage
) -> UIImage? {
  let width = baseImage.width
  let height = baseImage.height

  guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
  guard
    let context = CGContext(
      data: nil,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )
  else {
    return nil
  }

  let baseRect = CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height))
  context.draw(baseImage, in: baseRect)

  context.saveGState()
  context.setAlpha(0.5)
  context.draw(maskImage, in: baseRect)
  context.restoreGState()

  guard let composedImage = context.makeImage() else { return UIImage(cgImage: baseImage) }
  return UIImage(cgImage: composedImage)
}

public func drawYOLOClassifications(on ciImage: CIImage, result: YOLOResult) -> UIImage {
  guard let top5 = result.probs?.top5Labels else { return UIImage(ciImage: ciImage) }

  return renderWithBackground(ciImage) { ctx, size in
    let fontSize = max(size.width, size.height) / 50
    let labelMargin = fontSize / 2

    for (i, candidate) in top5.enumerated() {
      let colorIndex = (result.names.firstIndex(of: candidate) ?? 0) % ultralyticsColors.count
      let color = ultralyticsColors[colorIndex]
      let labelText = DetectionLabelStyle.text(
        className: candidate,
        confidence: CGFloat(result.probs?.top5Confs[i] ?? 0)
      )
      let textSize = DetectionLabelStyle.size(for: labelText, fontSize: fontSize)
      let labelRect = CGRect(
        x: labelMargin,
        y: labelMargin + (textSize.height + labelMargin) * CGFloat(i),
        width: textSize.width,
        height: textSize.height)

      ctx.setFillColor(color.cgColor)
      let labelPath = UIBezierPath(
        roundedRect: labelRect,
        cornerRadius: DetectionLabelStyle.cornerRadius
      )
      ctx.addPath(labelPath.cgPath)
      ctx.fillPath()
      let textPoint = CGPoint(
        x: labelRect.origin.x + DetectionLabelStyle.horizontalPadding / 2,
        y: labelRect.origin.y)
      labelText.draw(
        at: textPoint, withAttributes: DetectionLabelStyle.attributes(fontSize: fontSize))
    }
  } ?? UIImage()
}

extension UIColor {
  func toRGBComponents() -> (red: UInt8, green: UInt8, blue: UInt8)? {
    var red: CGFloat = 0
    var green: CGFloat = 0
    var blue: CGFloat = 0
    var alpha: CGFloat = 0

    let success = self.getRed(&red, green: &green, blue: &blue, alpha: &alpha)

    if success {
      let redUInt8 = UInt8(red * 255.0)
      let greenUInt8 = UInt8(green * 255.0)
      let blueUInt8 = UInt8(blue * 255.0)
      return (red: redUInt8, green: greenUInt8, blue: blueUInt8)
    } else {
      return nil
    }
  }
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
  let _radius = max(originalImageSize.width, originalImageSize.height) / 300
  for (i, keypoints) in keypointsList.enumerated() {
    drawSinglePersonKeypoints(
      keypoints: keypoints, confs: confsList[i], boundingBox: boundingBoxes[i],
      on: layer,
      imageViewSize: imageViewSize,
      originalImageSize: originalImageSize,
      radius: _radius,
      confThreshold: confThreshold,
      drawSkeleton: drawSkeleton
    )
  }
}

func drawSinglePersonKeypoints(
  keypoints: [(x: Float, y: Float)],
  confs: [Float],
  boundingBox: Box,
  on layer: CALayer,
  imageViewSize: CGSize,
  originalImageSize: CGSize,
  radius: CGFloat,
  confThreshold: Float,
  drawSkeleton: Bool
) {
  let lineWidth = radius * 0.4

  // Dynamic keypoint count support
  let numKeypoints = keypoints.count
  var points: [(CGPoint, Float)] = Array(repeating: (CGPoint.zero, 0), count: numKeypoints)

  for i in 0..<numKeypoints {
    let x = keypoints[i].x * Float(imageViewSize.width)
    let y = keypoints[i].y * Float(imageViewSize.height)
    let conf = confs[i]

    let point = CGPoint(x: CGFloat(x), y: CGFloat(y))
    let box = boundingBox

    if conf >= confThreshold
      && box.xywhn.contains(CGPoint(x: CGFloat(keypoints[i].x), y: CGFloat(keypoints[i].y)))
    {
      points[i] = (point, conf)

      // Use modulo to cycle through available colors for any number of keypoints
      let colorIndex = i < kptColorIndices.count ? kptColorIndices[i] : i % posePalette.count
      drawCircle(on: layer, at: point, radius: radius, color: colorIndex)
    }
  }

  if drawSkeleton {
    // Only draw skeleton if we have the standard 17 keypoints
    // For other keypoint counts, skeleton connectivity would need model-specific configuration
    if numKeypoints == 17 {
      for (index, bone) in skeleton.enumerated() {
        let (startIdx, endIdx) = (bone[0] - 1, bone[1] - 1)

        guard startIdx < points.count, endIdx < points.count else {
          continue
        }

        let startPoint = points[startIdx].0
        let endPoint = points[endIdx].0
        let startConf = points[startIdx].1
        let endConf = points[endIdx].1

        if startConf >= confThreshold && endConf >= confThreshold {
          let limbColorIndex =
            index < limbColorIndices.count ? limbColorIndices[index] : index % posePalette.count
          drawLine(
            on: layer, from: startPoint, to: endPoint, color: limbColorIndex,
            lineWidth: lineWidth)
        }
      }
    }
  }
}

func drawCircle(on layer: CALayer, at point: CGPoint, radius: CGFloat, color index: Int) {
  let circleLayer = CAShapeLayer()
  circleLayer.path =
    UIBezierPath(
      arcCenter: point,
      radius: radius,
      startAngle: 0,
      endAngle: .pi * 2,
      clockwise: true
    ).cgPath

  let color = posePalette[index].map { $0 / 255.0 }
  circleLayer.fillColor =
    UIColor(red: color[0], green: color[1], blue: color[2], alpha: 1.0).cgColor

  layer.addSublayer(circleLayer)
}

func drawLine(
  on layer: CALayer, from start: CGPoint, to end: CGPoint, color index: Int, lineWidth: CGFloat = 2
) {
  let lineLayer = CAShapeLayer()
  let path = UIBezierPath()
  path.move(to: start)
  path.addLine(to: end)

  lineLayer.path = path.cgPath
  // Ensure minimum line width for visibility
  lineLayer.lineWidth = max(lineWidth, 1.5)

  let color = posePalette[index].map { $0 / 255.0 }
  lineLayer.strokeColor =
    UIColor(red: color[0], green: color[1], blue: color[2], alpha: 1.0).cgColor

  layer.addSublayer(lineLayer)
}

func drawPoseOnCIImage(
  ciImage: CIImage,
  keypointsList: [[(x: Float, y: Float)]],
  confsList: [[Float]],
  boundingBoxes: [Box],
  originalImageSize: CGSize,
  radius: CGFloat = 5,
  confThreshold: Float = 0.25,
  drawSkeleton: Bool = true
) -> UIImage? {
  renderWithBackground(ciImage) { ctx, size in
    for box in boundingBoxes {
      drawBoxLabel(box, in: ctx, imageSize: size, rounded: true)
    }
    let poseLayer = CALayer()
    poseLayer.frame = CGRect(origin: .zero, size: size)
    drawKeypoints(
      keypointsList: keypointsList,
      confsList: confsList,
      boundingBoxes: boundingBoxes,
      on: poseLayer,
      imageViewSize: size,
      originalImageSize: originalImageSize,
      radius: radius,
      confThreshold: confThreshold,
      drawSkeleton: drawSkeleton)
    poseLayer.render(in: ctx)
  }
}

func drawOBBsOnCIImage(
  ciImage: CIImage,
  obbDetections: [OBBResult],
  targetSize: CGSize? = nil
) -> UIImage? {
  renderWithBackground(ciImage, targetSize: targetSize) { ctx, size in
    let lineWidth: CGFloat = max(size.width, size.height) / 200
    let fontSize = max(size.width, size.height) / 50
    ctx.setLineWidth(lineWidth)

    for detection in obbDetections {
      let color = ultralyticsColors[detection.index % ultralyticsColors.count]
      ctx.setStrokeColor(color.cgColor)

      // Plugin's `toPolygon(in:)` returns normalized corners; scale them to output pixels here.
      let normalizedCorners = detection.box.toPolygon(in: size)
      let corners = normalizedCorners.map {
        CGPoint(x: $0.x * size.width, y: $0.y * size.height)
      }
      ctx.beginPath()
      for (i, corner) in corners.enumerated() {
        i == 0 ? ctx.move(to: corner) : ctx.addLine(to: corner)
      }
      ctx.closePath()
      ctx.strokePath()

      if let first = corners.first {
        drawDetectionLabel(
          DetectionLabelStyle.text(
            className: detection.cls,
            confidence: CGFloat(detection.confidence)
          ),
          in: ctx,
          fontSize: fontSize,
          color: color,
          alpha: 1,
          anchor: first,
          cornerRadius: DetectionLabelStyle.cornerRadius,
          imageSize: size
        )
      }
    }
  }
}

/// Renders segmentation masks plus rounded bounding boxes onto the source image.
public func drawYOLOSegmentationWithBoxes(
  ciImage: CIImage,
  boxes: [Box],
  maskImage: CGImage?
) -> UIImage? {
  renderWithBackground(ciImage) { ctx, size in
    if let maskImage = maskImage {
      ctx.saveGState()
      ctx.setAlpha(0.5)
      // Flip to match the background orientation applied by renderWithBackground.
      ctx.translateBy(x: 0, y: size.height)
      ctx.scaleBy(x: 1, y: -1)
      ctx.draw(maskImage, in: CGRect(origin: .zero, size: size))
      ctx.restoreGState()
    }
    for box in boxes {
      drawBoxLabel(box, in: ctx, imageSize: size, rounded: true)
    }
  }
}

/// Renders a semantic segmentation color map onto the source image.
func drawYOLOSemanticSegmentation(
  ciImage: CIImage,
  semanticMask: CGImage?
) -> UIImage? {
  renderWithBackground(ciImage) { ctx, size in
    if let semanticMask = semanticMask {
      ctx.saveGState()
      ctx.setAlpha(0.5)
      ctx.translateBy(x: 0, y: size.height)
      ctx.scaleBy(x: 1, y: -1)
      ctx.draw(semanticMask, in: CGRect(origin: .zero, size: size))
      ctx.restoreGState()
    }
  }
}
