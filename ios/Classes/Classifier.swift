// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Ultralytics YOLO Package, implementing image classification functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The Classifier class implements image classification using YOLO models. Unlike object detection
//  or segmentation, it focuses on identifying the primary subject of an image rather than locating
//  objects within it. The class processes model outputs to extract classification probabilities,
//  identifying the top predicted class and confidence score. It supports multiple output formats
//  from Vision framework requests, handling both VNCoreMLFeatureValueObservation and
//  VNClassificationObservation result types. The implementation extracts both the top prediction
//  and the top 5 predictions with their confidence scores, enabling rich user feedback.

import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO classification models that identify the subject of an image.
class Classifier: BasePredictor, @unchecked Sendable {

  override var imageCropAndScaleOption: VNImageCropAndScaleOption { .centerCrop }

  /// Numerically stable softmax (max-subtraction) converting raw class logits to probabilities that sum to 1.
  static func softmax(_ logits: [Double]) -> [Double] {
    guard let maxLogit = logits.max() else { return logits }
    let exps = logits.map { exp($0 - maxLogit) }
    let sum = exps.reduce(0, +)
    guard sum > 0 else { return logits }
    return exps.map { $0 / sum }
  }

  override func setConfidenceThreshold(confidence: Double) {
    confidenceThreshold = confidence
    detector.featureProvider = ThresholdProvider(
      iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
  }

  override func setIouThreshold(iou: Double) {
    iouThreshold = iou
    detector.featureProvider = ThresholdProvider(
      iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
  }

  override func processObservations(for request: VNRequest, error: Error?) {
    let imageWidth = inputSize.width
    let imageHeight = inputSize.height
    self.inputSize = CGSize(width: imageWidth, height: imageHeight)
    var probs = Probs(top1Label: "", top5Labels: [], top1Conf: 0, top5Confs: [])

    if let observation = request.results as? [VNCoreMLFeatureValueObservation] {

      let multiArray = observation.first?.featureValue.multiArrayValue

      if let multiArray = multiArray {
        var rawValues = [Double]()
        for i in 0..<multiArray.count {
          rawValues.append(multiArray[i].doubleValue)
        }
        // Ultralytics `-cls` CoreML models emit raw logits; apply softmax so the reported confidences are real
        // probabilities in [0,1] (matches yolo-ios-app Classifier). Without it the overlay showed nonsense like
        // "cat 873%". Argmax ordering is unchanged since softmax is monotonic.
        let valuesArray = Self.softmax(rawValues)

        var indexedMap = [Int: Double]()
        for (index, value) in valuesArray.enumerated() {
          indexedMap[index] = value
        }

        let sortedMap = indexedMap.sorted { $0.value > $1.value }

        // top1
        if let (topIndex, topScore) = sortedMap.first {
          let top1Label = labelName(for: topIndex)
          let top1Conf = Float(topScore)
          probs.top1Label = top1Label
          probs.top1Conf = top1Conf
        }

        // top5
        let topObservations = sortedMap.prefix(5)
        var top5Labels: [String] = []
        var top5Confs: [Float] = []

        for (index, value) in topObservations {
          top5Labels.append(labelName(for: index))
          top5Confs.append(Float(value))
        }

        probs.top5Labels = top5Labels
        probs.top5Confs = top5Confs
      }
    } else if let observations = request.results as? [VNClassificationObservation] {
      var top1 = ""
      var top1Conf: Float = 0
      var top5: [String] = []
      var top5Confs: [Float] = []

      let candidateNumber = min(5, observations.count)
      if let topObservation = observations.first {
        top1 = topObservation.identifier
        top1Conf = Float(topObservation.confidence)
      }
      for i in 0..<candidateNumber {
        let observation = observations[i]
        let label = observation.identifier
        let confidence: Float = Float(observation.confidence)
        top5Confs.append(confidence)
        top5.append(label)
      }
      probs = Probs(top1Label: top1, top5Labels: top5, top1Conf: top1Conf, top5Confs: top5Confs)
    }

    let timing = updateTiming()
    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], probs: probs, speed: timing.speed, fps: timing.fps,
      names: labels)

    if let originalImageData = self.originalImageData {
      result.originalImage = UIImage(data: originalImageData)

    }

    self.currentOnResultsListener?.on(result: result)

  }

  override func predictOnImage(image: CIImage) -> YOLOResult {
    let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    guard let request = visionRequest else {
      let emptyResult = YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
      return emptyResult
    }

    let imageWidth = image.extent.width
    let imageHeight = image.extent.height
    self.inputSize = CGSize(width: imageWidth, height: imageHeight)
    var probs = Probs(top1Label: "", top5Labels: [], top1Conf: 0, top5Confs: [])
    do {
      try requestHandler.perform([request])
      if let observation = request.results as? [VNCoreMLFeatureValueObservation] {
        _ = [[String: Any]]()

        let multiArray = observation.first?.featureValue.multiArrayValue

        if let multiArray = multiArray {
          var rawValues = [Double]()
          for i in 0..<multiArray.count {
            rawValues.append(multiArray[i].doubleValue)
          }
          // Softmax the raw logits so confidences are real probabilities (see processObservations).
          let valuesArray = Self.softmax(rawValues)

          var indexedMap = [Int: Double]()
          for (index, value) in valuesArray.enumerated() {
            indexedMap[index] = value
          }

          let sortedMap = indexedMap.sorted { $0.value > $1.value }

          // top1
          if let (topIndex, topScore) = sortedMap.first {
            let top1Label = labelName(for: topIndex)
            let top1Conf = Float(topScore)
            probs.top1Label = top1Label
            probs.top1Conf = top1Conf
          }

          // top5
          let topObservations = sortedMap.prefix(5)
          var top5Labels: [String] = []
          var top5Confs: [Float] = []

          for (index, value) in topObservations {
            top5Labels.append(labelName(for: index))
            top5Confs.append(Float(value))
          }

          probs.top5Labels = top5Labels
          probs.top5Confs = top5Confs
        }
      } else if let observations = request.results as? [VNClassificationObservation] {
        var top1 = ""
        var top1Conf: Float = 0
        var top5: [String] = []
        var top5Confs: [Float] = []

        var candidateNumber = 5
        if observations.count < candidateNumber {
          candidateNumber = observations.count
        }
        if let topObservation = observations.first {
          top1 = topObservation.identifier
          top1Conf = Float(topObservation.confidence)
        }
        for i in 0..<candidateNumber {
          let observation = observations[i]
          let label = observation.identifier
          let confidence: Float = Float(observation.confidence)
          top5Confs.append(confidence)
          top5.append(label)
        }
        probs = Probs(top1Label: top1, top5Labels: top5, top1Conf: top1Conf, top5Confs: top5Confs)
      }

    } catch {
      NSLog("YOLO Classifier error: %@", String(describing: error))
    }

    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], probs: probs, speed: t1, names: labels)
    let annotatedImage = drawYOLOClassifications(on: image, result: result)
    result.annotatedImage = annotatedImage
    return result
  }
}
