// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Ultralytics YOLO Package, implementing instance segmentation functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The Segmenter class extends BasePredictor to provide instance segmentation capabilities.
//  Instance segmentation not only detects objects but also identifies the precise pixels
//  belonging to each object. The class processes complex model outputs including prototype masks
//  and detection results, performs non-maximum suppression to filter detections, and combines
//  results into visualizable mask images. It leverages the Accelerate framework for efficient
//  matrix operations and includes parallel processing to optimize performance on mobile devices.
//  The results include both bounding boxes and pixel-level masks that can be overlaid on images.

import Accelerate
import CoreML
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO segmentation models that identify objects and their pixel-level masks.
class Segmenter: BasePredictor, @unchecked Sendable {
  var colorsForMask: [(red: UInt8, green: UInt8, blue: UInt8)] = []

  override func setConfidenceThreshold(confidence: Double) {
    confidenceThreshold = confidence
  }

  override func setIouThreshold(iou: Double) {
    iouThreshold = iou
  }

  override func setNumItemsThreshold(numItems: Int) {
    numItemsThreshold = numItems
  }

  override func processObservations(for request: VNRequest, error: Error?) {
    if let results = request.results as? [VNCoreMLFeatureValueObservation] {
      //            DispatchQueue.main.async { [self] in
      guard results.count == 2 else { return }
      var pred: MLMultiArray
      var masks: MLMultiArray
      guard let out0 = results[0].featureValue.multiArrayValue,
        let out1 = results[1].featureValue.multiArrayValue
      else { return }
      let out0dim = checkShapeDimensions(of: out0)
      let out1dim = checkShapeDimensions(of: out1)
      if out0dim == 4 {
        masks = out0
        pred = out1
      } else {
        masks = out1
        pred = out0
      }
      let detectedObjects = postProcessSegment(
        feature: pred, confidenceThreshold: Float(confidenceThreshold),
        iouThreshold: Float(iouThreshold))
      var boxes: [Box] = []
      var alphas = [CGFloat]()

      // Apply numItemsThreshold limit
      let limitedDetections = Array(detectedObjects.prefix(numItemsThreshold))

      for p in limitedDetections {
        let box = p.0
        let confidence = p.2
        let bestClass = p.1
        let label = self.labelName(for: bestClass)
        let xywh = inputRect(fromModelRect: box)
        let rect = normalizedRect(fromInputRect: xywh)

        let boxResult = Box(index: bestClass, cls: label, conf: confidence, xywh: xywh, xywhn: rect)
        let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
        boxes.append(boxResult)
        alphas.append(alpha)
      }

      DispatchQueue.global(qos: .userInitiated).async {
        guard
          let processedMasks = generateCombinedMaskImage(
            detectedObjects: limitedDetections,
            protos: masks,
            inputWidth: self.modelInputSize.width,
            inputHeight: self.modelInputSize.height,
            threshold: 0.5,
            originalImageSize: self.inputSize

          ) as? (CGImage?, [[[Float]]])
        else {
          DispatchQueue.main.async { self.isUpdating = false }
          return
        }
        var maskResults = Masks(masks: processedMasks.1, combinedMask: processedMasks.0)

        let timing = self.updateTiming()

        var result = YOLOResult(
          orig_shape: self.inputSize, boxes: boxes, masks: maskResults, speed: timing.speed,
          fps: timing.fps, names: self.labels)

        if let originalImageData = self.originalImageData {
          result.originalImage = UIImage(data: originalImageData)

        }

        self.currentOnResultsListener?.on(result: result)
      }
    }
  }

  override func predictOnImage(image: CIImage) -> YOLOResult {
    let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    guard let request = visionRequest else {
      let emptyResult = YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
      return emptyResult
    }
    var boxes = [Box]()

    let imageWidth = image.extent.width
    let imageHeight = image.extent.height
    self.inputSize = CGSize(width: imageWidth, height: imageHeight)
    var result = YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)

    do {
      try requestHandler.perform([request])
      if let results = request.results as? [VNCoreMLFeatureValueObservation] {
        //                DispatchQueue.main.async { [self] in
        guard results.count == 2 else {
          return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)
        }
        var pred: MLMultiArray
        var masks: MLMultiArray
        guard let out0 = results[0].featureValue.multiArrayValue,
          let out1 = results[1].featureValue.multiArrayValue
        else { return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels) }
        let out0dim = checkShapeDimensions(of: out0)
        let out1dim = checkShapeDimensions(of: out1)
        if out0dim == 4 {
          masks = out0
          pred = out1
        } else {
          masks = out1
          pred = out0
        }
        let a = Date()

        let detectedObjects = postProcessSegment(
          feature: pred, confidenceThreshold: Float(confidenceThreshold),
          iouThreshold: Float(iouThreshold))
        var boxes: [Box] = []
        var colorMasks: [CGImage?] = []
        var alhaMasks: [CGImage?] = []
        var alphas = [CGFloat]()
        for p in detectedObjects {
          let box = p.0
          let confidence = p.2
          let bestClass = p.1
          let label = labelName(for: bestClass)
          let xywh = inputRect(fromModelRect: box)
          let rect = normalizedRect(fromInputRect: xywh)

          let boxResult = Box(
            index: bestClass, cls: label, conf: confidence, xywh: xywh, xywhn: rect)
          let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
          boxes.append(boxResult)
          alphas.append(alpha)
        }

        guard
          let processedMasks = generateCombinedMaskImage(
            detectedObjects: detectedObjects,
            protos: masks,
            inputWidth: self.modelInputSize.width,
            inputHeight: self.modelInputSize.height,
            threshold: 0.5,
            originalImageSize: self.inputSize

          ) as? (CGImage?, [[[Float]]])
        else {
          return YOLOResult(
            orig_shape: inputSize, boxes: boxes, masks: nil, annotatedImage: nil, speed: 0,
            names: labels)
        }
        guard
          let cgImage = CIContext().createCGImage(image, from: image.extent),
          let combinedMask = processedMasks.0
        else {
          return YOLOResult(
            orig_shape: inputSize, boxes: boxes, masks: nil, annotatedImage: nil, speed: 0,
            names: labels)
        }
        var annotatedImage = composeImageWithMask(
          baseImage: cgImage, maskImage: combinedMask)
        let maskResults = Masks(masks: processedMasks.1, combinedMask: combinedMask)
        let timing = updateTiming()
        result = YOLOResult(
          orig_shape: inputSize, boxes: boxes, masks: maskResults, annotatedImage: annotatedImage,
          speed: timing.speed, fps: timing.fps, names: labels)
        if let annotated = annotatedImage, let annotatedCI = CIImage(image: annotated) {
          annotatedImage = drawYOLODetections(on: annotatedCI, result: result)
          result.annotatedImage = annotatedImage
        }
        return result
      }
    } catch {
      NSLog("YOLO Segmenter: %@", String(describing: error))
    }
    return result
  }

  nonisolated func postProcessSegment(
    feature: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(CGRect, Int, Float, MLMultiArray)] {
    let shape = feature.shape.map { $0.intValue }
    guard shape.count == 3 else { return [] }
    if shape[2] < shape[1] {
      return postProcessEndToEndSegment(
        feature: feature, shape: shape, confidenceThreshold: confidenceThreshold)
    }

    let numAnchors = feature.shape[2].intValue
    let numFeatures = feature.shape[1].intValue
    let boxFeatureLength = 4
    let maskConfidenceLength = 32
    let numClasses = numFeatures - boxFeatureLength - maskConfidenceLength

    var results = [(CGRect, Int, Float, MLMultiArray)]()

    let featurePointer = feature.dataPointer.assumingMemoryBound(to: Float.self)
    let pointerWrapper = FloatPointerWrapper(featurePointer)

    let resultsQueue = DispatchQueue(label: "resultsQueue", attributes: .concurrent)

    DispatchQueue.concurrentPerform(iterations: numAnchors) { j in
      // Use pointerWrapper here
      let x = pointerWrapper.pointer[j]
      let y = pointerWrapper.pointer[numAnchors + j]
      let width = pointerWrapper.pointer[2 * numAnchors + j]
      let height = pointerWrapper.pointer[3 * numAnchors + j]

      let boxWidth = CGFloat(width)
      let boxHeight = CGFloat(height)
      let boxX = CGFloat(x - width / 2)
      let boxY = CGFloat(y - height / 2)

      let boundingBox = CGRect(x: boxX, y: boxY, width: boxWidth, height: boxHeight)

      // Class probabilities
      var classProbs = [Float](repeating: 0, count: numClasses)
      classProbs.withUnsafeMutableBufferPointer { classProbsPointer in
        vDSP_mtrans(
          pointerWrapper.pointer + 4 * numAnchors + j,
          numAnchors,
          classProbsPointer.baseAddress!,
          1,
          1,
          vDSP_Length(numClasses)
        )
      }
      var maxClassValue: Float = 0
      var maxClassIndex: vDSP_Length = 0
      vDSP_maxvi(classProbs, 1, &maxClassValue, &maxClassIndex, vDSP_Length(numClasses))

      if maxClassValue > confidenceThreshold {
        let maskProbsPointer = pointerWrapper.pointer + (4 + numClasses) * numAnchors + j
        let maskProbs = try! MLMultiArray(
          shape: [NSNumber(value: maskConfidenceLength)],
          dataType: .float32
        )
        for i in 0..<maskConfidenceLength {
          maskProbs[i] = NSNumber(value: maskProbsPointer[i * numAnchors])
        }

        let result = (boundingBox, Int(maxClassIndex), maxClassValue, maskProbs)

        resultsQueue.async(flags: .barrier) {
          results.append(result)
        }
      }
    }

    resultsQueue.sync(flags: .barrier) {}

    var selectedBoxesAndFeatures = [(CGRect, Int, Float, MLMultiArray)]()

    for classIndex in 0..<numClasses {
      let classResults = results.filter { $0.1 == classIndex }
      if !classResults.isEmpty {
        let boxesOnly = classResults.map { $0.0 }
        let scoresOnly = classResults.map { $0.2 }
        let selectedIndices = nonMaxSuppression(
          boxes: boxesOnly,
          scores: scoresOnly,
          threshold: iouThreshold
        )
        for idx in selectedIndices {
          selectedBoxesAndFeatures.append(
            (
              classResults[idx].0,
              classResults[idx].1,
              classResults[idx].2,
              classResults[idx].3
            )
          )
        }
      }
    }

    return selectedBoxesAndFeatures
  }

  private nonisolated func postProcessEndToEndSegment(
    feature: MLMultiArray,
    shape: [Int],
    confidenceThreshold: Float
  ) -> [(CGRect, Int, Float, MLMultiArray)] {
    let maskConfidenceLength = 32
    let strides = feature.strides.map { $0.intValue }
    let pointer = feature.dataPointer.assumingMemoryBound(to: Float.self)
    let detStride = strides[1]
    let fieldStride = strides[2]
    let maskStart = shape[2] > 5 ? 6 : 5
    var results = [(CGRect, Int, Float, MLMultiArray)]()

    for i in 0..<shape[1] {
      let base = i * detStride
      let confidence = pointer[base + 4 * fieldStride]
      guard confidence > confidenceThreshold else { continue }

      let x1 = CGFloat(pointer[base])
      let y1 = CGFloat(pointer[base + fieldStride])
      let x2 = CGFloat(pointer[base + 2 * fieldStride])
      let y2 = CGFloat(pointer[base + 3 * fieldStride])
      let classIndex = shape[2] > 5 ? Int(pointer[base + 5 * fieldStride]) : 0
      guard
        let maskProbs = try? MLMultiArray(
          shape: [NSNumber(value: maskConfidenceLength)], dataType: .float32)
      else { continue }

      let maskPointer = maskProbs.dataPointer.assumingMemoryBound(to: Float.self)
      for m in 0..<min(maskConfidenceLength, shape[2] - maskStart) {
        maskPointer[m] = pointer[base + (maskStart + m) * fieldStride]
      }

      results.append(
        (
          CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1), classIndex, confidence,
          maskProbs
        ))
    }

    return results
  }

  func checkShapeDimensions(of multiArray: MLMultiArray) -> Int {
    let shapeAsInts = multiArray.shape.map { $0.intValue }
    let dimensionCount = shapeAsInts.count

    return dimensionCount
  }

}

final class FloatPointerWrapper: @unchecked Sendable {
  let pointer: UnsafeMutablePointer<Float>
  init(_ pointer: UnsafeMutablePointer<Float>) {
    self.pointer = pointer
  }
}
