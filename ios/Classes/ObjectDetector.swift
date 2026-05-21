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
    detector.featureProvider = ThresholdProvider(
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
    detector.featureProvider = ThresholdProvider(
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
    let boxes = decodeBoxes(from: request)
    let timing = updateTiming()
    var result = YOLOResult(
      orig_shape: inputSize, boxes: boxes, speed: timing.speed, fps: timing.fps, names: labels)

    if let originalImageData = self.originalImageData {
      result.originalImage = UIImage(data: originalImageData)
    }

    self.currentOnResultsListener?.on(result: result)
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
      boxes = decodeBoxes(from: request)
    } catch {
      NSLog("YOLO ObjectDetector error: %@", String(describing: error))
    }
    let speed = Date().timeIntervalSince(start)
    var result = YOLOResult(orig_shape: inputSize, boxes: boxes, speed: speed, names: labels)
    let annotatedImage = drawYOLODetections(on: image, result: result)
    result.annotatedImage = annotatedImage

    return result
  }

  private func decodeBoxes(from request: VNRequest) -> [Box] {
    if let results = request.results as? [VNRecognizedObjectObservation] {
      return decodeRecognizedBoxes(results)
    }
    if let results = request.results as? [VNCoreMLFeatureValueObservation],
      let prediction = results.first?.featureValue.multiArrayValue
    {
      return decodeEndToEndBoxes(prediction)
    }
    return []
  }

  private func decodeRecognizedBoxes(_ results: [VNRecognizedObjectObservation]) -> [Box] {
    var boxes = [Box]()
    boxes.reserveCapacity(min(results.count, numItemsThreshold))
    for prediction in results.prefix(numItemsThreshold) {
      guard let topLabel = prediction.labels.first else { continue }
      let invertedBox = CGRect(
        x: prediction.boundingBox.minX, y: 1 - prediction.boundingBox.maxY,
        width: prediction.boundingBox.width, height: prediction.boundingBox.height)
      let imageRect = VNImageRectForNormalizedRect(
        invertedBox, Int(inputSize.width), Int(inputSize.height))
      let label = topLabel.identifier
      boxes.append(
        Box(
          index: labels.firstIndex(of: label) ?? 0, cls: label, conf: topLabel.confidence,
          xywh: imageRect, xywhn: invertedBox))
    }
    return boxes
  }

  private func decodeEndToEndBoxes(_ prediction: MLMultiArray) -> [Box] {
    let shape = prediction.shape.map { $0.intValue }
    guard shape.count == 3, shape[2] < shape[1], shape[2] >= 6 else { return [] }

    let strides = prediction.strides.map { $0.intValue }
    let pointer = prediction.dataPointer.assumingMemoryBound(to: Float.self)
    let detStride = strides[1]
    let fieldStride = strides[2]
    var boxes = [Box]()

    for i in 0..<shape[1] {
      let base = i * detStride
      let confidence = pointer[base + 4 * fieldStride]
      guard confidence > Float(confidenceThreshold) else { continue }

      let x1 = CGFloat(pointer[base])
      let y1 = CGFloat(pointer[base + fieldStride])
      let x2 = CGFloat(pointer[base + 2 * fieldStride])
      let y2 = CGFloat(pointer[base + 3 * fieldStride])
      let classIndex = Int(pointer[base + 5 * fieldStride])
      let imageRect = inputRect(
        fromModelRect: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1))
      let label = labelName(for: classIndex)

      boxes.append(
        Box(
          index: classIndex, cls: label, conf: confidence, xywh: imageRect,
          xywhn: normalizedRect(fromInputRect: imageRect)))
      if boxes.count >= numItemsThreshold { break }
    }
    return boxes
  }
}
