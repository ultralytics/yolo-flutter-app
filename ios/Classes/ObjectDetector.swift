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
    if let results = request.results as? [VNRecognizedObjectObservation] {
      var boxes = [Box]()

      // Limit the number of detections based on numItemsThreshold
      let maxDetections = min(results.count, self.numItemsThreshold)

      for i in 0..<maxDetections {
        let prediction = results[i]
        guard let topLabel = prediction.labels.first else { continue }
        let invertedBox = CGRect(
          x: prediction.boundingBox.minX, y: 1 - prediction.boundingBox.maxY,
          width: prediction.boundingBox.width, height: prediction.boundingBox.height)
        let imageRect: CGRect
        if modelInputSize.width > 0, modelInputSize.height > 0 {
          let modelRect = CGRect(
            x: invertedBox.minX * CGFloat(modelInputSize.width),
            y: invertedBox.minY * CGFloat(modelInputSize.height),
            width: invertedBox.width * CGFloat(modelInputSize.width),
            height: invertedBox.height * CGFloat(modelInputSize.height))
          imageRect = inputRect(fromModelRect: modelRect)
        } else {
          imageRect = VNImageRectForNormalizedRect(
            invertedBox, Int(inputSize.width), Int(inputSize.height))
        }
        let normalizedBox = normalizedRect(fromInputRect: imageRect)

        let label = topLabel.identifier
        let index = self.labels.firstIndex(of: label) ?? 0
        let confidence = topLabel.confidence
        let box = Box(
          index: index, cls: label, conf: confidence, xywh: imageRect, xywhn: normalizedBox)
        boxes.append(box)
      }

      let timing = updateTiming()
      var result = YOLOResult(
        orig_shape: inputSize, boxes: boxes, speed: timing.speed, fps: timing.fps, names: labels)

      if let originalImageData = self.originalImageData {
        result.originalImage = UIImage(data: originalImageData)

      }

      self.currentOnResultsListener?.on(result: result)

    }
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
        // Limit the number of detections based on numItemsThreshold
        let maxDetections = min(results.count, self.numItemsThreshold)

        for i in 0..<maxDetections {
          let prediction = results[i]
          guard let topLabel = prediction.labels.first else { continue }
          let invertedBox = CGRect(
            x: prediction.boundingBox.minX, y: 1 - prediction.boundingBox.maxY,
            width: prediction.boundingBox.width, height: prediction.boundingBox.height)
          let imageRect: CGRect
          if modelInputSize.width > 0, modelInputSize.height > 0 {
            let modelRect = CGRect(
              x: invertedBox.minX * CGFloat(modelInputSize.width),
              y: invertedBox.minY * CGFloat(modelInputSize.height),
              width: invertedBox.width * CGFloat(modelInputSize.width),
              height: invertedBox.height * CGFloat(modelInputSize.height))
            imageRect = inputRect(fromModelRect: modelRect)
          } else {
            imageRect = VNImageRectForNormalizedRect(
              invertedBox, Int(inputSize.width), Int(inputSize.height))
          }
          let normalizedBox = normalizedRect(fromInputRect: imageRect)

          let label = topLabel.identifier
          let index = self.labels.firstIndex(of: label) ?? 0
          let confidence = topLabel.confidence
          let box = Box(
            index: index, cls: label, conf: confidence, xywh: imageRect, xywhn: normalizedBox)
          boxes.append(box)
        }
      }
    } catch {
      NSLog("YOLO ObjectDetector error: %@", String(describing: error))
    }
    let speed = Date().timeIntervalSince(start)
    var result = YOLOResult(orig_shape: inputSize, boxes: boxes, speed: t1, names: labels)
    let annotatedImage = drawYOLODetections(on: image, result: result)
    result.annotatedImage = annotatedImage

    return result
  }
}
