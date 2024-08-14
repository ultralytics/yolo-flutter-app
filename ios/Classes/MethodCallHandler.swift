//
//  MethodChannelCallHandler.swift
//  ultralytics_yolo
//
//  Created by Sergio Sánchez on 9/11/23.
//

import Foundation

class MethodCallHandler: VideoCaptureDelegate, InferenceTimeListener, ResultsListener,
  FpsRateListener
{
  private let resultStreamHandler: ResultStreamHandler
  private let inferenceTimeStreamHandler: TimeStreamHandler
  private let fpsRateStreamHandler: TimeStreamHandler
  private var predictor: Predictor?
  private let videoCapture: VideoCapture

  init(binaryMessenger: FlutterBinaryMessenger, videoCapture: VideoCapture) {
    resultStreamHandler = ResultStreamHandler()
    let resultsEventChannel = FlutterEventChannel(
      name: "ultralytics_yolo_prediction_results", binaryMessenger: binaryMessenger)
    resultsEventChannel.setStreamHandler(resultStreamHandler)

    let inferenceTimeEventChannel = FlutterEventChannel(
      name: "ultralytics_yolo_inference_time", binaryMessenger: binaryMessenger)
    inferenceTimeStreamHandler = TimeStreamHandler()
    inferenceTimeEventChannel.setStreamHandler(inferenceTimeStreamHandler)

    let fpsRateEventChannel = FlutterEventChannel(
      name: "ultralytics_yolo_fps_rate", binaryMessenger: binaryMessenger)
    fpsRateStreamHandler = TimeStreamHandler()
    fpsRateEventChannel.setStreamHandler(fpsRateStreamHandler)

    self.videoCapture = videoCapture
    videoCapture.delegate = self
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    guard let args = call.arguments as? [String: Any] else {
      return
    }
    if call.method == "loadModel" {
      Task {
        await loadModel(args: args, result: result)
      }
    } else if call.method == "setConfidenceThreshold" {
      setConfidenceThreshold(args: args, result: result)
    } else if call.method == "setIouThreshold" {
      setIouThreshold(args: args, result: result)
    } else if call.method == "setNumItemsThreshold" {
      setNumItemsThreshold(args: args, result: result)
    } else if call.method == "setLensDirection" {
      setLensDirection(args: args, result: result)
    } else if call.method == "closeCamera" {
      closeCamera(args: args, result: result)
    } else if call.method == "detectImage" || call.method == "classifyImage" {
      predictOnImage(args: args, result: result)
    }
  }

  public func videoCapture(
    _ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer
  ) {
    predictor?.predict(
      sampleBuffer: sampleBuffer, onResultsListener: self, onInferenceTime: self, onFpsRate: self)
  }

  private func loadModel(args: [String: Any], result: @escaping FlutterResult) async {
    let flutterError = FlutterError(
      code: "PredictorError",
      message: "Invalid model",
      details: nil)

    guard let model = args["model"] as? [String: Any] else {
      result(flutterError)
      return
    }
    guard let type = model["type"] as? String else {
      result(flutterError)
      return
    }

    var yoloModel: (any YoloModel)?
    guard let task = model["task"] as? String
    else {
      result(flutterError)
      return
    }

    switch type {
    case "local":
      guard let modelPath = model["modelPath"] as? String
      else {
        result(flutterError)
        return
      }
      yoloModel = LocalModel(modelPath: modelPath, task: task)
      break
    case "remote":
      break
    default:
      result(flutterError)
      return
    }

    do {
      switch task {
      case "detect":
        predictor = try await ObjectDetector(yoloModel: yoloModel!)
        break
      case "classify":
        predictor = try await ObjectClassifier(yoloModel: yoloModel!)
        break
      default:
        result(flutterError)
        return
      }

      result("Success")
    } catch {
      result(flutterError)
    }
  }

  private func setConfidenceThreshold(args: [String: Any], result: @escaping FlutterResult) {
    let conf = args["confidence"] as! Double
    (predictor as? ObjectDetector)?.setConfidenceThreshold(confidence: conf)
  }

  private func setIouThreshold(args: [String: Any], result: @escaping FlutterResult) {
    let iou = args["iou"] as! Double
    (predictor as? ObjectDetector)?.setIouThreshold(iou: iou)
  }

  private func setNumItemsThreshold(args: [String: Any], result: @escaping FlutterResult) {
    let numItems = args["numItems"] as! Int
    (predictor as? ObjectDetector)?.setNumItemsThreshold(numItems: numItems)
  }

  private func setLensDirection(args: [String: Any], result: @escaping FlutterResult) {
    let direction = args["direction"] as? Int

    //        startCameraPreview(position: direction == 0 ? .back : .front)
  }

  private func closeCamera(args: [String: Any], result: @escaping FlutterResult) {
    videoCapture.stop()
  }

  private func createCIImage(fromPath path: String) throws -> CIImage? {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    return CIImage(data: data)
  }

  private func predictOnImage(args: [String: Any], result: @escaping FlutterResult) {
    let imagePath = args["imagePath"] as! String
    let image = try? createCIImage(fromPath: imagePath)
    predictor?.predictOnImage(
      image: image!,
      completion: { recognitions in
        result(recognitions)
      })
  }

  func on(predictions: [[String: Any]]) {
    resultStreamHandler.sink(objects: predictions)
  }

  func on(inferenceTime: Double) {
    inferenceTimeStreamHandler.sink(time: inferenceTime)
  }

  func on(fpsRate: Double) {
    fpsRateStreamHandler.sink(time: fpsRate)
  }
}
