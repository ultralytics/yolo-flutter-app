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
        feature: pred,
        masks: masks,
        confidenceThreshold: Float(confidenceThreshold),
        iouThreshold: Float(iouThreshold))
      var boxes: [Box] = []
      var alphas = [CGFloat]()

      // Apply numItemsThreshold limit
      let limitedDetections = Array(detectedObjects.prefix(numItemsThreshold))

      for p in limitedDetections {
        let box = p.0
        let inputW = max(1.0, CGFloat(self.modelInputSize.width))
        let inputH = max(1.0, CGFloat(self.modelInputSize.height))
        let rect = CGRect(
          x: box.minX / inputW,
          y: box.minY / inputH,
          width: box.width / inputW,
          height: box.height / inputH)
        let confidence = p.2
        let bestClass = p.1
        let label = self.labels[bestClass]
        let xywh = VNImageRectForNormalizedRect(
          rect, Int(self.inputSize.width), Int(self.inputSize.height))

        let boxResult = Box(index: bestClass, cls: label, conf: confidence, xywh: xywh, xywhn: rect)
        let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
        boxes.append(boxResult)
        alphas.append(alpha)
      }

      DispatchQueue.global(qos: .userInitiated).async {
        guard
          let procceessedMasks = generateCombinedMaskImage(
            detectedObjects: limitedDetections,
            protos: masks,
            inputWidth: self.modelInputSize.width,
            inputHeight: self.modelInputSize.height,
            threshold: 0.5

          ) as? (CGImage?, [[[Float]]])
        else {
          DispatchQueue.main.async { self.isUpdating = false }
          return
        }
        var maskResults = Masks(masks: procceessedMasks.1, combinedMask: procceessedMasks.0)
        var result = YOLOResult(
          orig_shape: self.inputSize, boxes: boxes, masks: maskResults, speed: self.t2,
          fps: 1 / self.t4, names: self.labels)

        if let originalImageData = self.originalImageData {
          result.originalImage = UIImage(data: originalImageData)

        }

        self.updateTime()
        self.currentOnResultsListener?.on(result: result)
      }
    }
  }

  private func updateTime() {
    if self.t1 < 10.0 {  // valid dt
      self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
    }
    self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
    self.t3 = CACurrentMediaTime()

    self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)  // t2 seconds to ms

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
          feature: pred,
          masks: masks,
          confidenceThreshold: Float(self.confidenceThreshold),
          iouThreshold: Float(self.iouThreshold))
        var boxes: [Box] = []
        var colorMasks: [CGImage?] = []
        var alhaMasks: [CGImage?] = []
        var alphas = [CGFloat]()
        let limitedDetections = Array(detectedObjects.prefix(self.numItemsThreshold))
        for p in limitedDetections {
          let box = p.0
          let inputW = max(1.0, CGFloat(self.modelInputSize.width))
          let inputH = max(1.0, CGFloat(self.modelInputSize.height))
          let rect = CGRect(
            x: box.minX / inputW,
            y: box.minY / inputH,
            width: box.width / inputW,
            height: box.height / inputH)
          let confidence = p.2
          let bestClass = p.1
          let label = labels[bestClass]
          let xywh = VNImageRectForNormalizedRect(rect, Int(inputSize.width), Int(inputSize.height))

          let boxResult = Box(
            index: bestClass, cls: label, conf: confidence, xywh: xywh, xywhn: rect)
          let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
          boxes.append(boxResult)
          alphas.append(alpha)
        }

        guard
          let procceessedMasks = generateCombinedMaskImage(
            detectedObjects: limitedDetections,
            protos: masks,
            inputWidth: self.modelInputSize.width,
            inputHeight: self.modelInputSize.height,
            threshold: 0.5

          ) as? (CGImage?, [[[Float]]])
        else {
          return YOLOResult(
            orig_shape: inputSize, boxes: boxes, masks: nil, annotatedImage: nil, speed: 0,
            names: labels)
        }
        let cgImage = CIContext().createCGImage(image, from: image.extent)!
        var annotatedImage = composeImageWithMask(
          baseImage: cgImage, maskImage: procceessedMasks.0!)
        var maskResults: Masks = Masks(masks: procceessedMasks.1, combinedMask: procceessedMasks.0)
        if self.t1 < 10.0 {  // valid dt
          self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
        }
        self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
        self.t3 = CACurrentMediaTime()
        result = YOLOResult(
          orig_shape: inputSize, boxes: boxes, masks: maskResults, annotatedImage: annotatedImage,
          speed: self.t2, fps: 1 / self.t4, names: labels)
        annotatedImage = drawYOLODetections(on: CIImage(image: annotatedImage!)!, result: result)
        result.annotatedImage = annotatedImage
        return result

        //                }
      }
    } catch {
      print(error)
    }
    return result
  }

  nonisolated func postProcessSegment(
    feature: MLMultiArray,
    masks: MLMultiArray?,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(CGRect, Int, Float, MLMultiArray)] {
    let maskChannels = maskChannelCount(from: masks)
    if isYOLO26Model {
      return postProcessYOLO26Segment(
        prediction: feature,
        maskChannels: maskChannels,
        confidenceThreshold: confidenceThreshold,
        iouThreshold: iouThreshold)
    } else {
      return postProcessLegacySegment(
        feature: feature,
        maskChannels: maskChannels,
        confidenceThreshold: confidenceThreshold,
        iouThreshold: iouThreshold)
    }
  }

  private func postProcessLegacySegment(
    feature: MLMultiArray,
    maskChannels: Int,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(CGRect, Int, Float, MLMultiArray)] {
    let numAnchors = feature.shape[2].intValue
    let numFeatures = feature.shape[1].intValue
    let boxFeatureLength = 4
    let numClasses = max(0, numFeatures - boxFeatureLength - maskChannels)

    var results = [(CGRect, Int, Float, MLMultiArray)]()

    let featurePointer = feature.dataPointer.assumingMemoryBound(to: Float.self)
    let pointerWrapper = FloatPointerWrapper(featurePointer)

    let resultsQueue = DispatchQueue(label: "resultsQueue", attributes: .concurrent)

    DispatchQueue.concurrentPerform(iterations: numAnchors) { j in
      let x = pointerWrapper.pointer[j]
      let y = pointerWrapper.pointer[numAnchors + j]
      let width = pointerWrapper.pointer[2 * numAnchors + j]
      let height = pointerWrapper.pointer[3 * numAnchors + j]

      let boxWidth = CGFloat(width)
      let boxHeight = CGFloat(height)
      let boxX = CGFloat(x - width / 2)
      let boxY = CGFloat(y - height / 2)

      let boundingBox = CGRect(x: boxX, y: boxY, width: boxWidth, height: boxHeight)

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
          shape: [NSNumber(value: maskChannels)],
          dataType: .float32
        )
        for i in 0..<maskChannels {
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

  private func postProcessYOLO26Segment(
    prediction: MLMultiArray,
    maskChannels: Int,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(CGRect, Int, Float, MLMultiArray)] {
    let shape = prediction.shape.map { $0.intValue }
    guard shape.count >= 2 else { return [] }

    let numDetections: Int
    let stride: Int
    if shape.count == 3 {
      numDetections = shape[1]
      stride = shape[2]
    } else {
      numDetections = shape[0]
      stride = shape[1]
    }

    guard stride >= 6 else {
      print("YOLO26 Segment: invalid stride \(stride) for shape \(shape)")
      return []
    }

    let ptr = prediction.dataPointer.assumingMemoryBound(to: Float.self)
    let modelW = CGFloat(max(1, modelInputSize.width))
    let modelH = CGFloat(max(1, modelInputSize.height))

    var rawDetections: [(CGRect, Int, Float, MLMultiArray)] = []
    rawDetections.reserveCapacity(min(numDetections, 200))

    for i in 0..<numDetections {
      let off = i * stride
      let x1 = CGFloat(ptr[off + 0])
      let y1 = CGFloat(ptr[off + 1])
      let x2 = CGFloat(ptr[off + 2])
      let y2 = CGFloat(ptr[off + 3])
      let conf = normalizeYOLOScore(ptr[off + 4])

      guard conf > confidenceThreshold else { continue }

      let clsIndex = Int(round(ptr[off + 5]))

      let boxX = x1
      let boxY = y1
      let boxW = x2 - x1
      let boxH = y2 - y1

      guard boxW > 0, boxH > 0 else { continue }

      let normX = max(0.0, min(1.0, boxX / modelW))
      let normY = max(0.0, min(1.0, boxY / modelH))
      let normW = max(0.0, min(1.0 - normX, boxW / modelW))
      let normH = max(0.0, min(1.0 - normY, boxH / modelH))

      guard normW > 0.0, normH > 0.0 else { continue }

      let boundingBox = CGRect(x: boxX, y: boxY, width: boxW, height: boxH)

      let coeffCount = max(0, min(maskChannels, stride - 6))
      let maskCoeffs = try! MLMultiArray(shape: [NSNumber(value: coeffCount)], dataType: .float32)
      for k in 0..<coeffCount {
        maskCoeffs[k] = NSNumber(value: ptr[off + 6 + k])
      }

      rawDetections.append((boundingBox, clsIndex, conf, maskCoeffs))
    }

    if rawDetections.isEmpty { return [] }

    let boxesOnly = rawDetections.map { $0.0 }
    let scoresOnly = rawDetections.map { $0.2 }
    let selectedIdx = nonMaxSuppression(
      boxes: boxesOnly, scores: scoresOnly, threshold: iouThreshold)

    return selectedIdx.map { rawDetections[$0] }
  }

  func adjustBox(_ box: CGRect, toFitIn containerSize: CGSize) -> CGRect {
    let xScale = containerSize.width / 640.0
    let yScale = containerSize.height / 640.0
    return CGRect(
      x: box.origin.x * xScale, y: box.origin.y * yScale, width: box.size.width * xScale,
      height: box.size.height * yScale)
  }

  func checkShapeDimensions(of multiArray: MLMultiArray) -> Int {
    let shapeAsInts = multiArray.shape.map { $0.intValue }
    let dimensionCount = shapeAsInts.count

    return dimensionCount
  }

  private func maskChannelCount(from masks: MLMultiArray?) -> Int {
    guard let masks = masks else { return 32 }
    if masks.shape.count >= 2 {
      if masks.shape.count >= 3 {
        return masks.shape[1].intValue
      } else {
        return masks.shape[0].intValue
      }
    }
    return 32
  }

}

final class FloatPointerWrapper: @unchecked Sendable {
  let pointer: UnsafeMutablePointer<Float>
  init(_ pointer: UnsafeMutablePointer<Float>) {
    self.pointer = pointer
  }
}
