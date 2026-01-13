// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Ultralytics YOLO Package, implementing object detection functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The ObjectDetector class provides specialized functionality for detecting objects in images
//  using YOLO models. It processes Vision framework results to extract bounding boxes, class labels,
//  and confidence scores from model predictions. The class handles both real-time frame processing
//  and single image analysis, converting the Vision API's normalized coordinates to image coordinates,
//  and packaging the results in the standardized YOLOResult format. It includes performance monitoring
//  for inference time and frame rate, and offers runtime adjustable parameters such as confidence
//  threshold and IoU threshold for non-maximum suppression.

import CoreML
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO object detection models that identifies and localizes objects in images.
///
/// This class processes the outputs from YOLO object detection models, extracting bounding boxes,
/// class labels, and confidence scores. It handles both real-time camera feed processing and
/// single image analysis, converting the normalized coordinates from the Vision framework
/// to image coordinates and applying non-maximum suppression to filter duplicative detections.
///
/// - Note: Object detection models output rectangular bounding boxes around detected objects.
/// - SeeAlso: `Segmenter` for models that produce pixel-level masks for objects.
class ObjectDetector: BasePredictor {

  /// Sets the confidence threshold and updates the model's feature provider.
  ///
  /// This overridden method ensures that when the confidence threshold is changed,
  /// the Vision model's feature provider is also updated to use the new value.
  ///
  /// - Parameter confidence: The new confidence threshold value (0.0 to 1.0).
  override func setConfidenceThreshold(confidence: Double) {
    confidenceThreshold = confidence
    detector?.featureProvider = ThresholdProvider(
      iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
  }

  /// Sets the IoU threshold and updates the model's feature provider.
  ///
  /// This overridden method ensures that when the IoU threshold is changed,
  /// the Vision model's feature provider is also updated to use the new value.
  ///
  /// - Parameter iou: The new IoU threshold value (0.0 to 1.0).
  override func setIouThreshold(iou: Double) {
    iouThreshold = iou
    detector?.featureProvider = ThresholdProvider(
      iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
  }

  /// Processes the results from the Vision framework's object detection request.
  ///
  /// This method extracts bounding boxes, class labels, and confidence scores from the
  /// Vision object detection results, converts coordinates to the original image space,
  /// and notifies listeners with the structured detection results.
  ///
  /// - Parameters:
  ///   - request: The completed Vision request containing object detection results.
  ///   - error: Any error that occurred during the Vision request.
  override func processObservations(for request: VNRequest, error: Error?) {
    if let error = error {
      print("ObjectDetector error: \(error.localizedDescription)")
      return
    }

    guard let results = request.results else {
      return
    }

    if let results = results as? [VNRecognizedObjectObservation] {
      var boxes = [Box]()

      let maxDetections = min(results.count, self.numItemsThreshold)

      for i in 0..<maxDetections {
        let prediction = results[i]
        let invertedBox = CGRect(
          x: prediction.boundingBox.minX, y: 1 - prediction.boundingBox.maxY,
          width: prediction.boundingBox.width, height: prediction.boundingBox.height)
        let imageRect = VNImageRectForNormalizedRect(
          invertedBox, Int(inputSize.width), Int(inputSize.height))

        let label = prediction.labels[0].identifier
        let index = self.labels.firstIndex(of: label) ?? 0
        let confidence = prediction.labels[0].confidence
        let box = Box(
          index: index, cls: label, conf: confidence, xywh: imageRect, xywhn: invertedBox)
        boxes.append(box)
      }

      if self.t1 < 10.0 {
        self.t2 = self.t1 * 0.05 + self.t2 * 0.95
      }
      self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95
      self.t3 = CACurrentMediaTime()

      self.currentOnInferenceTimeListener?.on(
        inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)
      var result = YOLOResult(
        orig_shape: inputSize, boxes: boxes, speed: self.t2, fps: 1 / self.t4, names: labels)

      if let originalImageData = self.originalImageData {
        result.originalImage = UIImage(data: originalImageData)
      }

      self.currentOnResultsListener?.on(result: result)
    } else if let featureResults = results as? [VNCoreMLFeatureValueObservation] {
      guard let prediction = featureResults.first?.featureValue.multiArrayValue else {
        print("ObjectDetector: No MLMultiArray in feature results")
        return
      }

      let detectedObjects = postProcessDetection(
        feature: prediction,
        confidenceThreshold: Float(self.confidenceThreshold),
        iouThreshold: Float(self.iouThreshold)
      )

      var boxes: [Box] = []
      let inputWidth = Int(inputSize.width)
      let inputHeight = Int(inputSize.height)

      let limitedObjects = detectedObjects.prefix(self.numItemsThreshold)
      for detection in limitedObjects {
        let (box, classIndex, confidence) = detection
        let rect = CGRect(
          x: box.minX,
          y: box.minY,
          width: box.width,
          height: box.height
        )
        let label = (classIndex < labels.count) ? labels[classIndex] : "unknown"
        let xywh = VNImageRectForNormalizedRect(rect, inputWidth, inputHeight)

        let boxResult = Box(
          index: classIndex,
          cls: label,
          conf: confidence,
          xywh: xywh,
          xywhn: rect
        )
        boxes.append(boxResult)
      }

      if self.t1 < 10.0 {
        self.t2 = self.t1 * 0.05 + self.t2 * 0.95
      }
      self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95
      self.t3 = CACurrentMediaTime()

      let result = YOLOResult(
        orig_shape: inputSize,
        boxes: boxes,
        speed: self.t2,
        fps: 1 / self.t4,
        names: labels
      )

      self.currentOnInferenceTimeListener?.on(
        inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)
      var mutableResult = result
      if let originalImageData = self.originalImageData {
        mutableResult.originalImage = UIImage(data: originalImageData)
      }
      self.currentOnResultsListener?.on(result: mutableResult)
    }
  }


  private func postProcessDetection(
    feature: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(CGRect, Int, Float)] {
    let shape = feature.shape.map { $0.intValue }

    if isYOLO26Model && shape.count >= 2 && shape.last == 6 {
      return postProcessYOLO26Format(
        feature: feature, shape: shape, confidenceThreshold: confidenceThreshold)
    }

    var numAnchors: Int
    var numFeatures: Int

    if shape.count == 3 {
      numAnchors = shape[1]
      numFeatures = shape[2]
    } else if shape.count == 2 {
      numAnchors = shape[0]
      numFeatures = shape[1]
    } else {
      print("ObjectDetector: Unexpected feature shape: \(shape)")
      return []
    }

    let boxFeatureLength = 4
    let numClasses = numFeatures - boxFeatureLength

    guard numClasses > 0 else {
      print("ObjectDetector: Invalid number of classes: \(numClasses)")
      return []
    }

    let featurePointer = feature.dataPointer.assumingMemoryBound(to: Float.self)

    var detections: [(CGRect, Int, Float)] = []
    detections.reserveCapacity(min(numAnchors / 10, 100))

    func sigmoid(_ x: Float) -> Float {
      return 1.0 / (1.0 + exp(-x))
    }

    var sampleScores: [Float] = []
    let sampleCount = min(10, numAnchors)
    for i in 0..<sampleCount {
      let offset = i * numFeatures
      for c in 0..<min(3, numClasses) {
        let score = featurePointer[offset + boxFeatureLength + c]
        sampleScores.append(score)
      }
    }
    let maxSample = sampleScores.max() ?? 0
    let minSample = sampleScores.min() ?? 0
    let needsNormalization = maxSample > 10.0 || minSample < -10.0

    for i in 0..<numAnchors {
      let offset = i * numFeatures
      let cx = CGFloat(featurePointer[offset])
      let cy = CGFloat(featurePointer[offset + 1])
      let w = CGFloat(featurePointer[offset + 2])
      let h = CGFloat(featurePointer[offset + 3])

      let boxX = cx - w / 2
      let boxY = cy - h / 2
      let box = CGRect(x: boxX, y: boxY, width: w, height: h)

      var bestScore: Float = 0
      var bestClass: Int = 0
      for c in 0..<numClasses {
        var score = featurePointer[offset + boxFeatureLength + c]

        if isYOLO26Model {
          score = normalizeYOLOScore(score)
        }

        if score > bestScore {
          bestScore = score
          bestClass = c
        }
      }

      var normalizedScore = normalizeYOLOScore(bestScore)
      if !isYOLO26Model && needsNormalization {
        normalizedScore = sigmoid(bestScore)
      }

      if normalizedScore > confidenceThreshold && normalizedScore <= 1.0 {
        detections.append((box, bestClass, normalizedScore))
      }
    }

    if isYOLO26Model {
      detections.sort { $0.2 > $1.2 }
      return detections
    }

    var classBuckets: [Int: [(CGRect, Int, Float)]] = [:]
    for detection in detections {
      let classIndex = detection.1
      if classBuckets[classIndex] == nil {
        classBuckets[classIndex] = []
      }
      classBuckets[classIndex]?.append(detection)
    }

    var selectedDetections: [(CGRect, Int, Float)] = []
    for (_, classDetections) in classBuckets {
      let boxesOnly = classDetections.map { $0.0 }
      let scoresOnly = classDetections.map { $0.2 }
      let selectedIndices = nonMaxSuppression(
        boxes: boxesOnly,
        scores: scoresOnly,
        threshold: iouThreshold
      )
      for idx in selectedIndices {
        selectedDetections.append(classDetections[idx])
      }
    }

    selectedDetections.sort { $0.2 > $1.2 }

    return selectedDetections
  }

 
  private func postProcessYOLO26Format(
    feature: MLMultiArray,
    shape: [Int],
    confidenceThreshold: Float
  ) -> [(CGRect, Int, Float)] {
    let numDetections: Int
    if shape.count == 3 {
      numDetections = shape[1]
    } else if shape.count == 2 {
      numDetections = shape[0]
    } else {
      print(
        "ObjectDetector: Invalid YOLO26 format, expected [1, num_detections, 6] or [num_detections, 6], got \(shape)"
      )
      return []
    }

    let featurePointer = feature.dataPointer.assumingMemoryBound(to: Float.self)
    var detections: [(CGRect, Int, Float)] = []

    let stride: Int
    if shape.count == 3 {
      stride = shape[2]
    } else {
      stride = shape[1]
    }

    let modelWidth = CGFloat(self.modelInputSize.width)
    let modelHeight = CGFloat(self.modelInputSize.height)

    for i in 0..<numDetections {
      let offset = i * stride

      let x1 = CGFloat(featurePointer[offset])
      let y1 = CGFloat(featurePointer[offset + 1])
      let x2 = CGFloat(featurePointer[offset + 2])
      let y2 = CGFloat(featurePointer[offset + 3])
      var confidence = featurePointer[offset + 4]
      let classIndex = Int(round(featurePointer[offset + 5]))

      confidence = normalizeYOLOScore(confidence)

      var boxX: CGFloat = 0
      var boxY: CGFloat = 0
      var boxW: CGFloat = 0
      var boxH: CGFloat = 0

      if modelWidth > 0 && modelHeight > 0 {
        boxX = x1 / modelWidth
        boxY = y1 / modelHeight
        boxW = (x2 - x1) / modelWidth
        boxH = (y2 - y1) / modelHeight
      } else {
        boxX = x1
        boxY = y1
        boxW = x2 - x1
        boxH = y2 - y1
      }

      boxX = max(0.0, min(1.0, boxX))
      boxY = max(0.0, min(1.0, boxY))
      boxW = max(0.0, min(1.0 - boxX, boxW))
      boxH = max(0.0, min(1.0 - boxY, boxH))

      let box = CGRect(x: boxX, y: boxY, width: boxW, height: boxH)

      let isValidBox = boxW > 0.01 && boxH > 0.01 && boxW <= 1.0 && boxH <= 1.0
      let hasValidConfidence = confidence > confidenceThreshold && confidence <= 1.0
      let hasValidClass = classIndex >= 0 && classIndex < labels.count

      if isValidBox && hasValidConfidence && hasValidClass {
        detections.append((box, classIndex, confidence))
      }
    }

    detections.sort { $0.2 > $1.2 }

    return detections
  }

  /// Processes a static image and returns object detection results.
  ///
  /// This method performs object detection on a static image and returns the
  /// detection results synchronously. It handles the entire inference pipeline
  /// from setting up the Vision request to processing the detection results.
  ///
  /// - Parameter image: The CIImage to analyze for object detection.
  /// - Returns: A YOLOResult containing the detected objects with bounding boxes, class labels, and confidence scores.
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
    let start = Date()

    do {
      try requestHandler.perform([request])
      if let results = request.results as? [VNRecognizedObjectObservation] {
        let maxDetections = min(results.count, self.numItemsThreshold)

        for i in 0..<maxDetections {
          let prediction = results[i]
          let invertedBox = CGRect(
            x: prediction.boundingBox.minX, y: 1 - prediction.boundingBox.maxY,
            width: prediction.boundingBox.width, height: prediction.boundingBox.height)
          let imageRect = VNImageRectForNormalizedRect(
            invertedBox, Int(inputSize.width), Int(inputSize.height))

          let label = prediction.labels[0].identifier
          let index = self.labels.firstIndex(of: label) ?? 0
          let confidence = prediction.labels[0].confidence
          let box = Box(
            index: index, cls: label, conf: confidence, xywh: imageRect, xywhn: invertedBox)
          boxes.append(box)
        }
      } else if let featureResults = request.results as? [VNCoreMLFeatureValueObservation],
                let prediction = featureResults.first?.featureValue.multiArrayValue {
        let detections = postProcessDetection(
          feature: prediction,
          confidenceThreshold: Float(self.confidenceThreshold),
          iouThreshold: Float(self.iouThreshold)
        )

        let inputWidth = Int(inputSize.width)
        let inputHeight = Int(inputSize.height)

        for detection in detections.prefix(self.numItemsThreshold) {
          let (box, classIndex, confidence) = detection
          let rect = CGRect(
            x: box.minX,
            y: box.minY,
            width: box.width,
            height: box.height
          )
          let label = (classIndex < labels.count) ? labels[classIndex] : "unknown"
          let xywh = VNImageRectForNormalizedRect(rect, inputWidth, inputHeight)

          let boxResult = Box(
            index: classIndex,
            cls: label,
            conf: confidence,
            xywh: xywh,
            xywhn: rect
          )
          boxes.append(boxResult)
        }
      }
    } catch {
      print(error)
    }
    let speed = Date().timeIntervalSince(start)
    var result = YOLOResult(orig_shape: inputSize, boxes: boxes, speed: speed, names: labels)
    let annotatedImage = drawYOLODetections(on: image, result: result)
    result.annotatedImage = annotatedImage

    return result
  }
}
