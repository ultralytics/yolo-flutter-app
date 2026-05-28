// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Ultralytics YOLO Package, implementing human pose estimation functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The PoseEstimater class extends the BasePredictor to provide human pose and keypoint detection.
//  It processes model outputs to identify human subjects and their body keypoints (joints such as
//  eyes, shoulders, elbows, wrists, hips, knees, ankles, etc.). The class converts the model's raw
//  output into structured data representing each detected person's bounding box and associated
//  keypoints with their confidence scores. This implementation supports both real-time processing
//  for camera feeds and single image analysis, producing visualizable results that can be overlaid
//  on the source image to show the detected pose skeleton.

import Accelerate
import CoreML
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO pose estimation models that identify human body keypoints.
class PoseEstimater: BasePredictor, @unchecked Sendable {
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

      if let prediction = results.first?.featureValue.multiArrayValue {

        let preds = PostProcessPose(
          prediction: prediction, confidenceThreshold: Float(self.confidenceThreshold),
          iouThreshold: Float(self.iouThreshold))
        var keypointsList = [Keypoints]()
        var boxes = [Box]()

        // Apply numItemsThreshold limit
        let limitedPreds = Array(preds.prefix(numItemsThreshold))

        for person in limitedPreds {
          boxes.append(person.box)
          keypointsList.append(person.keypoints)
        }
        let timing = updateTiming()

        var result = YOLOResult(
          orig_shape: inputSize, boxes: boxes, masks: nil, probs: nil, keypointsList: keypointsList,
          annotatedImage: nil, speed: timing.speed, fps: timing.fps, originalImage: nil,
          names: labels)

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
    var _: [Box] = []

    let imageWidth = image.extent.width
    let imageHeight = image.extent.height
    self.inputSize = CGSize(width: imageWidth, height: imageHeight)
    let result = YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)

    do {
      try requestHandler.perform([request])

      if let results = request.results as? [VNCoreMLFeatureValueObservation] {

        if let prediction = results.first?.featureValue.multiArrayValue {

          let preds = PostProcessPose(
            prediction: prediction, confidenceThreshold: Float(self.confidenceThreshold),
            iouThreshold: Float(self.iouThreshold))
          var keypointsList = [Keypoints]()
          var boxes = [Box]()
          var keypointsForImage = [[(x: Float, y: Float)]]()
          var confsList: [[Float]] = []

          // Apply numItemsThreshold limit
          let limitedPreds = Array(preds.prefix(numItemsThreshold))

          for person in limitedPreds {
            boxes.append(person.box)
            keypointsList.append(person.keypoints)
            keypointsForImage.append(person.keypoints.xyn)
            confsList.append(person.keypoints.conf)
          }

          let annotatedImage = drawPoseOnCIImage(
            ciImage: image, keypointsList: keypointsForImage, confsList: confsList,
            boundingBoxes: boxes, originalImageSize: inputSize)
          let timing = updateTiming()
          return YOLOResult(
            orig_shape: inputSize, boxes: boxes, masks: nil, probs: nil,
            keypointsList: keypointsList, annotatedImage: annotatedImage, speed: timing.speed,
            fps: timing.fps, originalImage: nil, names: labels)
        }
      }
    } catch {
      NSLog("YOLO PoseEstimator error: %@", String(describing: error))
    }
    return result
  }

  func PostProcessPose(
    prediction: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  )
    -> [(box: Box, keypoints: Keypoints)]
  {
    let shape = prediction.shape.map { $0.intValue }
    guard shape.count == 3 else { return [] }
    if shape[2] < shape[1] {
      return postProcessEndToEndPose(
        prediction, shape: shape, confidenceThreshold: confidenceThreshold)
    }

    let numAnchors = prediction.shape[2].intValue
    let featureCount = prediction.shape[1].intValue - 5

    var boxes = [CGRect]()
    var scores = [Float]()
    var features = [[Float]]()

    let featurePointer = UnsafeMutablePointer<Float>(OpaquePointer(prediction.dataPointer))
    let lock = DispatchQueue(label: "com.example.lock")

    DispatchQueue.concurrentPerform(iterations: numAnchors) { j in
      let confIndex = 4 * numAnchors + j
      let confidence = featurePointer[confIndex]

      if confidence > confidenceThreshold {
        let x = featurePointer[j]
        let y = featurePointer[numAnchors + j]
        let width = featurePointer[2 * numAnchors + j]
        let height = featurePointer[3 * numAnchors + j]

        let boxWidth = CGFloat(width)
        let boxHeight = CGFloat(height)
        let boxX = CGFloat(x - width / 2.0)
        let boxY = CGFloat(y - height / 2.0)
        let boundingBox = CGRect(
          x: boxX, y: boxY,
          width: boxWidth, height: boxHeight)

        var boxFeatures = [Float](repeating: 0, count: featureCount)
        for k in 0..<featureCount {
          let key = (5 + k) * numAnchors + j
          boxFeatures[k] = featurePointer[key]
        }

        lock.sync {
          boxes.append(boundingBox)
          scores.append(confidence)
          features.append(boxFeatures)
        }
      }
    }

    let selectedIndices = nonMaxSuppression(boxes: boxes, scores: scores, threshold: iouThreshold)

    let filteredBoxes = selectedIndices.map { boxes[$0] }
    let filteredScores = selectedIndices.map { scores[$0] }
    let filteredFeatures = selectedIndices.map { features[$0] }

    let boxScorePairs = zip(filteredBoxes, filteredScores)
    let results: [(Box, Keypoints)] = zip(boxScorePairs, filteredFeatures).map {
      (pair, boxFeatures) in
      let (box, score) = pair
      let imageSizeBox = inputRect(fromModelRect: box)
      let normalizedBox = normalizedRect(fromInputRect: imageSizeBox)
      let boxResult = Box(
        index: 0, cls: "person", conf: score, xywh: imageSizeBox, xywhn: normalizedBox)
      let numKeypoints = boxFeatures.count / 3

      var xynArray = [(x: Float, y: Float)]()
      var xyArray = [(x: Float, y: Float)]()
      var confArray = [Float]()

      for i in 0..<numKeypoints {
        let kx = boxFeatures[3 * i]
        let ky = boxFeatures[3 * i + 1]
        let kc = boxFeatures[3 * i + 2]

        let imagePoint = inputPoint(fromModelPoint: CGPoint(x: CGFloat(kx), y: CGFloat(ky)))
        let pointNorm = normalizedPoint(fromInputPoint: imagePoint)
        xynArray.append((x: Float(pointNorm.x), y: Float(pointNorm.y)))
        xyArray.append((x: Float(imagePoint.x), y: Float(imagePoint.y)))

        confArray.append(kc)
      }

      let keypoints = Keypoints(xyn: xynArray, xy: xyArray, conf: confArray)
      return (boxResult, keypoints)
    }

    return results
  }

  private func postProcessEndToEndPose(
    _ prediction: MLMultiArray,
    shape: [Int],
    confidenceThreshold: Float
  ) -> [(box: Box, keypoints: Keypoints)] {
    let strides = prediction.strides.map { $0.intValue }
    let pointer = prediction.dataPointer.assumingMemoryBound(to: Float.self)
    let detStride = strides[1]
    let fieldStride = strides[2]
    let keypointStart = (shape[2] - 6) % 3 == 0 ? 6 : 5
    let keypointCount = (shape[2] - keypointStart) / 3
    var results = [(box: Box, keypoints: Keypoints)]()

    for i in 0..<shape[1] {
      let base = i * detStride
      let confidence = pointer[base + 4 * fieldStride]
      guard confidence > confidenceThreshold else { continue }

      let x1 = CGFloat(pointer[base])
      let y1 = CGFloat(pointer[base + fieldStride])
      let x2 = CGFloat(pointer[base + 2 * fieldStride])
      let y2 = CGFloat(pointer[base + 3 * fieldStride])
      let imageSizeBox = inputRect(
        fromModelRect: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1))
      let boxResult = Box(
        index: 0, cls: "person", conf: confidence, xywh: imageSizeBox,
        xywhn: normalizedRect(fromInputRect: imageSizeBox))

      var xynArray = [(x: Float, y: Float)]()
      var xyArray = [(x: Float, y: Float)]()
      var confArray = [Float]()

      for k in 0..<keypointCount {
        let keypointBase = base + (keypointStart + 3 * k) * fieldStride
        let imagePoint = inputPoint(
          fromModelPoint: CGPoint(
            x: CGFloat(pointer[keypointBase]),
            y: CGFloat(pointer[keypointBase + fieldStride])))
        let pointNorm = normalizedPoint(fromInputPoint: imagePoint)
        xynArray.append((x: Float(pointNorm.x), y: Float(pointNorm.y)))
        xyArray.append((x: Float(imagePoint.x), y: Float(imagePoint.y)))
        confArray.append(pointer[keypointBase + 2 * fieldStride])
      }

      results.append((boxResult, Keypoints(xyn: xynArray, xy: xyArray, conf: confArray)))
      if results.count >= numItemsThreshold { break }
    }

    return results
  }
}
