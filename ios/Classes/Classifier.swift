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
class Classifier: BasePredictor {

  private var isYOLO26Model: Bool {
    guard let path = modelURL?.lastPathComponent.lowercased() else { return false }
    return path.contains("yolo26")
  }

  private func parseProbs(from multiArray: MLMultiArray) -> Probs {
    var values = [Double](repeating: 0, count: multiArray.count)
    for i in 0..<multiArray.count {
      values[i] = multiArray[i].doubleValue
    }

    let maxLogit = values.max() ?? 0
    let expValues = values.map { exp($0 - maxLogit) }
    let sumExp = expValues.reduce(0, +)
    let probsArray = sumExp > 0 ? expValues.map { $0 / sumExp } : values

    var indexedMap = [Int: Double]()
    for (index, value) in probsArray.enumerated() {
      indexedMap[index] = value
    }

    let sortedMap = indexedMap.sorted { $0.value > $1.value }

    var result = Probs(top1Label: "", top5Labels: [], top1Conf: 0, top5Confs: [])

    if let (topIndex, topScore) = sortedMap.first, labels.indices.contains(topIndex) {
      result.top1Label = labels[topIndex]
      result.top1Conf = Float(topScore)
    }

    let topObservations = sortedMap.prefix(5)
    for (index, value) in topObservations where labels.indices.contains(index) {
      result.top5Labels.append(labels[index])
      result.top5Confs.append(Float(value))
    }

    return result
  }

  override func setConfidenceThreshold(confidence: Double) {
    confidenceThreshold = confidence
    detector?.featureProvider = ThresholdProvider(
      iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
  }

  override func setIouThreshold(iou: Double) {
    iouThreshold = iou
    detector?.featureProvider = ThresholdProvider(
      iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
  }

  override func processObservations(for request: VNRequest, error: Error?) {
    let imageWidth = inputSize.width
    let imageHeight = inputSize.height
    self.inputSize = CGSize(width: imageWidth, height: imageHeight)
    var probs = Probs(top1Label: "", top5Labels: [], top1Conf: 0, top5Confs: [])

    if let observation = request.results as? [VNCoreMLFeatureValueObservation],
      let multiArray = observation.first?.featureValue.multiArrayValue
    {
      probs = parseProbs(from: multiArray)
    } else if let observations = request.results as? [VNClassificationObservation] {
      var top1 = ""
      var top1Conf: Float = 0
      var top5: [String] = []
      var top5Confs: [Float] = []

      var candidateNumber = min(5, observations.count)
      if let topObservation = observations.first {
        top1 = topObservation.identifier
        top1Conf = Float(topObservation.confidence)
      }
      for i in 0...candidateNumber - 1 {
        let observation = observations[i]
        let label = observation.identifier
        let confidence: Float = Float(observation.confidence)
        top5Confs.append(confidence)
        top5.append(label)
      }
      probs = Probs(top1Label: top1, top5Labels: top5, top1Conf: top1Conf, top5Confs: top5Confs)
    }

    if self.t1 < 10.0 {
      self.t2 = self.t1 * 0.05 + self.t2 * 0.95
    }
    self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95
    self.t3 = CACurrentMediaTime()

    self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)
    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], probs: probs, speed: self.t2, fps: 1 / self.t4,
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
      if let observation = request.results as? [VNCoreMLFeatureValueObservation],
        let multiArray = observation.first?.featureValue.multiArrayValue
      {
        probs = parseProbs(from: multiArray)
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
        for i in 0...candidateNumber - 1 {
          let observation = observations[i]
          let label = observation.identifier
          let confidence: Float = Float(observation.confidence)
          top5Confs.append(confidence)
          top5.append(label)
        }
        probs = Probs(top1Label: top1, top5Labels: top5, top1Conf: top1Conf, top5Confs: top5Confs)
      }

    } catch {
      print(error)
    }

    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], probs: probs, speed: t1, names: labels)
    let annotatedImage = drawYOLOClassifications(on: image, result: result)
    result.annotatedImage = annotatedImage
    return result
  }
}
