// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import Foundation
import UIKit
import Vision

/// Predictor for YOLO semantic segmentation models that output dense logits.
class SemanticSegmenter: BasePredictor, @unchecked Sendable {
  private var colorCache: (classCount: Int, colors: [(red: UInt8, green: UInt8, blue: UInt8)])?

  override func processObservations(for request: VNRequest, error: Error?) {
    let semanticMask = firstFeatureArray(request).flatMap { postProcessSemantic($0) }
    let timing = updateTiming()
    let result = YOLOResult(
      orig_shape: inputSize, boxes: [], semanticMask: semanticMask, speed: timing.speed,
      fps: timing.fps, names: labels)
    currentOnResultsListener?.on(result: result)
  }

  override func predictOnImage(image: CIImage) -> YOLOResult {
    let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    guard let request = visionRequest else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }

    inputSize = CGSize(width: image.extent.width, height: image.extent.height)
    let start = Date()
    var semanticMask: SemanticMask?

    do {
      try requestHandler.perform([request])
      semanticMask = firstFeatureArray(request).flatMap { postProcessSemantic($0) }
    } catch {
      NSLog("Semantic segmentation failed: %@", String(describing: error))
    }

    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], semanticMask: semanticMask,
      speed: Date().timeIntervalSince(start), names: labels)
    result.annotatedImage = drawYOLOSemanticSegmentation(
      ciImage: image, semanticMask: semanticMask?.maskImage)
    return result
  }

  private func firstFeatureArray(_ request: VNRequest) -> MLMultiArray? {
    (request.results as? [VNCoreMLFeatureValueObservation])?.first?.featureValue.multiArrayValue
  }

  func postProcessSemantic(_ logits: MLMultiArray) -> SemanticMask? {
    let shape = logits.shape.map { $0.intValue }
    let strides = logits.strides.map { $0.intValue }
    guard shape.count == 4, shape[0] == 1 else { return nil }

    let isNCHW = shape[1] <= shape[3] || shape[1] == labels.count
    let classCount = isNCHW ? shape[1] : shape[3]
    let maskHeight = isNCHW ? shape[2] : shape[1]
    let maskWidth = isNCHW ? shape[3] : shape[2]
    guard classCount > 0, maskWidth > 0, maskHeight > 0 else { return nil }

    let bounds = CGRect(x: 0, y: 0, width: maskWidth, height: maskHeight)
    let outputRect = (modelMaskCropRect(maskWidth: maskWidth, maskHeight: maskHeight) ?? bounds)
      .intersection(bounds).integral
    let outputX = Int(outputRect.minX)
    let outputY = Int(outputRect.minY)
    let outputWidth = Int(outputRect.width)
    let outputHeight = Int(outputRect.height)
    guard outputWidth > 0, outputHeight > 0 else { return nil }

    var classMap = [Int](repeating: 0, count: outputWidth * outputHeight)
    var pixels = [UInt8](repeating: 0, count: outputWidth * outputHeight * 4)
    // A single-channel mask is foreground/background: allocate 2 colors (0 = background, 1 = foreground) and threshold
    // each pixel, instead of painting the whole frame as class 0. Mirrors yolo-ios-app SemanticSegmenter.
    let colors = semanticColors(classCount: classCount == 1 ? 2 : classCount)
    let binaryThreshold: Float = classCount == 1 ? singleChannelThreshold(logits) : 0

    for y in 0..<outputHeight {
      let sourceY = y + outputY
      for x in 0..<outputWidth {
        let sourceX = x + outputX
        let classIndex = bestClass(
          logits: logits, strides: strides, classCount: classCount,
          x: sourceX, y: sourceY, isNCHW: isNCHW, binaryThreshold: binaryThreshold)
        let outputIndex = y * outputWidth + x
        classMap[outputIndex] = classIndex
        writeColor(colors[classIndex], into: &pixels, at: outputIndex * 4)
      }
    }

    return SemanticMask(
      classMap: classMap,
      width: outputWidth,
      height: outputHeight,
      maskImage: makeImage(fromRGBA: pixels, width: outputWidth, height: outputHeight))
  }

  /// Min/max scan to decide the binary cutoff: probability-like outputs (in [0,1]) threshold at 0.5, raw logits at 0.
  /// Mirrors yolo-ios-app SemanticSegmenter.singleChannelThreshold.
  private func singleChannelThreshold(_ logits: MLMultiArray) -> Float {
    var minValue = Float.greatestFiniteMagnitude
    var maxValue = -Float.greatestFiniteMagnitude
    for i in 0..<logits.count {
      let v = logits[i].floatValue
      minValue = min(minValue, v)
      maxValue = max(maxValue, v)
    }
    return minValue >= 0 && maxValue <= 1 ? 0.5 : 0
  }

  private func bestClass(
    logits: MLMultiArray,
    strides: [Int],
    classCount: Int,
    x: Int,
    y: Int,
    isNCHW: Bool,
    binaryThreshold: Float
  ) -> Int {
    if classCount == 1 {
      let score = value(
        in: logits, strides: strides, classIndex: 0, x: x, y: y, isNCHW: isNCHW)
      return score > binaryThreshold ? 1 : 0
    }

    var bestIndex = 0
    var bestScore = -Float.greatestFiniteMagnitude
    for classIndex in 0..<classCount {
      let score = value(
        in: logits, strides: strides, classIndex: classIndex, x: x, y: y, isNCHW: isNCHW)
      if score > bestScore {
        bestScore = score
        bestIndex = classIndex
      }
    }
    return bestIndex
  }

  private func value(
    in logits: MLMultiArray,
    strides: [Int],
    classIndex: Int,
    x: Int,
    y: Int,
    isNCHW: Bool
  ) -> Float {
    let offset =
      isNCHW
      ? classIndex * strides[1] + y * strides[2] + x * strides[3]
      : y * strides[1] + x * strides[2] + classIndex * strides[3]
    return value(in: logits, at: offset, classIndex: classIndex, x: x, y: y, isNCHW: isNCHW)
  }

  private func value(
    in logits: MLMultiArray,
    at offset: Int,
    classIndex: Int,
    x: Int,
    y: Int,
    isNCHW: Bool
  ) -> Float {
    switch logits.dataType {
    case .float32:
      return logits.dataPointer.assumingMemoryBound(to: Float.self)[offset]
    case .double:
      return Float(logits.dataPointer.assumingMemoryBound(to: Double.self)[offset])
    case .int32:
      return Float(logits.dataPointer.assumingMemoryBound(to: Int32.self)[offset])
    default:
      let indexes = isNCHW ? [0, classIndex, y, x] : [0, y, x, classIndex]
      return logits[indexes.map { NSNumber(value: $0) }].floatValue
    }
  }

  private func semanticColors(classCount: Int) -> [(red: UInt8, green: UInt8, blue: UInt8)] {
    if let colorCache, colorCache.classCount == classCount {
      return colorCache.colors
    }

    let colors = (0..<classCount).map { classIndex in
      let color = ultralyticsColors[classIndex % ultralyticsColors.count]
      var red: CGFloat = 0
      var green: CGFloat = 0
      var blue: CGFloat = 0
      color.getRed(&red, green: &green, blue: &blue, alpha: nil)
      return (UInt8(red * 255), UInt8(green * 255), UInt8(blue * 255))
    }
    colorCache = (classCount, colors)
    return colors
  }

  private func writeColor(
    _ color: (red: UInt8, green: UInt8, blue: UInt8), into pixels: inout [UInt8], at offset: Int
  ) {
    pixels[offset] = color.red
    pixels[offset + 1] = color.green
    pixels[offset + 2] = color.blue
    pixels[offset + 3] = 255
  }

  private func makeImage(fromRGBA pixels: [UInt8], width: Int, height: Int) -> CGImage? {
    let data = Data(pixels)
    guard let provider = CGDataProvider(data: data as CFData) else { return nil }
    return CGImage(
      width: width,
      height: height,
      bitsPerComponent: 8,
      bitsPerPixel: 32,
      bytesPerRow: width * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
      provider: provider,
      decode: nil,
      shouldInterpolate: true,
      intent: .defaultIntent)
  }
}
