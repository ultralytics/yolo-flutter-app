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

import Accelerate
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO classification models that identify the subject of an image.
class Classifier: BasePredictor, @unchecked Sendable {

  override var imageCropAndScaleOption: VNImageCropAndScaleOption { .centerCrop }

  override func setConfidenceThreshold(confidence: Double) {
    confidenceThreshold = confidence
    // Honor requiresNMS (IoU 1.0 for NMS-free models) so this setter doesn't clobber the create()-time seed.
    detector.featureProvider = ThresholdProvider(
      iouThreshold: requiresNMS ? iouThreshold : 1.0, confidenceThreshold: confidenceThreshold)
  }

  override func setIouThreshold(iou: Double) {
    iouThreshold = iou
    detector.featureProvider = ThresholdProvider(
      iouThreshold: requiresNMS ? iouThreshold : 1.0, confidenceThreshold: confidenceThreshold)
  }

  override func processObservations(for request: VNRequest, error: Error?) {
    let probs = extractProbs(from: request)
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
    guard let request = visionRequest else {
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }

    var probs = Probs(top1Label: "", top5Labels: [], top1Conf: 0, top5Confs: [])
    let requestHandler = makeRequestHandler(for: image)
    if perform(request, with: requestHandler, errorMessage: "YOLO Classifier error") {
      probs = extractProbs(from: request)
    }

    var result = YOLOResult(
      orig_shape: inputSize, boxes: [], probs: probs, speed: 0, names: labels)
    result.annotatedImage = drawYOLOClassifications(on: image, result: result)
    result.speed = finishTiming(notify: false)
    return result
  }

  /// Extracts top-1 and top-5 probabilities from a Vision request result, handling both
  /// `VNCoreMLFeatureValueObservation` (raw logits requiring softmax) and `VNClassificationObservation` (already
  /// normalized scores). Mirrors yolo-ios-app Classifier.
  private func extractProbs(from request: VNRequest) -> Probs {
    if let observations = request.results as? [VNCoreMLFeatureValueObservation],
      let multiArray = observations.first?.featureValue.multiArrayValue
    {
      return softmaxProbs(from: multiArray)
    }
    if let observations = request.results as? [VNClassificationObservation] {
      let top = observations.prefix(5)
      return Probs(
        top1Label: observations.first?.identifier ?? "",
        top5Labels: top.map { $0.identifier },
        top1Conf: Float(observations.first?.confidence ?? 0),
        top5Confs: top.map { Float($0.confidence) }
      )
    }
    return Probs(top1Label: "", top5Labels: [], top1Conf: 0, top5Confs: [])
  }

  /// Applies a numerically stable softmax to raw class logits and returns the top-1/top-5 probabilities.
  ///
  /// Ultralytics `-cls` CoreML models emit raw logits; softmax turns them into real probabilities in [0,1] (without
  /// it the overlay showed values like "cat 873%"). Uses Accelerate (vDSP) for the softmax and a single linear pass
  /// with a tiny sorted insertion buffer for the top-5 — this avoids the O(n log n) sort and the per-frame
  /// tuple-array allocation the previous implementation paid on every classification. Argmax ordering is unchanged.
  func softmaxProbs(from multiArray: MLMultiArray) -> Probs {
    let count = multiArray.count
    var logits = [Float](repeating: 0, count: count)
    if multiArray.dataType == .float32, multiArray.strides.last?.intValue == 1 {
      let src = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
      logits.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: src, count: count) }
    } else {
      for i in 0..<count { logits[i] = multiArray[i].floatValue }
    }

    var output = [Float](repeating: 0, count: count)
    var maxLogit: Float = 0
    vDSP_maxv(logits, 1, &maxLogit, vDSP_Length(count))
    var negMax = -maxLogit
    vDSP_vsadd(logits, 1, &negMax, &output, 1, vDSP_Length(count))
    var n = Int32(count)
    vvexpf(&output, output, &n)
    var sum: Float = 0
    vDSP_sve(output, 1, &sum, vDSP_Length(count))
    if sum > 0 {
      vDSP_vsdiv(output, 1, &sum, &output, 1, vDSP_Length(count))
    }

    // Top-5 via one linear pass into a small sorted buffer. Equal scores resolve to the lower class index.
    let k = min(5, count)
    var topIdx = [Int](repeating: -1, count: k)
    var topVal = [Float](repeating: -.greatestFiniteMagnitude, count: k)
    for i in 0..<count {
      let v = output[i]
      if v <= topVal[k - 1] { continue }
      var p = k - 1
      while p > 0 && v > topVal[p - 1] {
        topVal[p] = topVal[p - 1]
        topIdx[p] = topIdx[p - 1]
        p -= 1
      }
      topVal[p] = v
      topIdx[p] = i
    }
    var topLabels = [String]()
    var topConfs = [Float]()
    for j in 0..<k where topIdx[j] >= 0 && topIdx[j] < labels.count {
      topLabels.append(labels[topIdx[j]])
      topConfs.append(topVal[j])
    }
    return Probs(
      top1Label: topLabels.first ?? "",
      top5Labels: topLabels,
      top1Conf: topConfs.first ?? 0,
      top5Confs: topConfs
    )
  }
}
