//
//  MethodChannelCallHandler.swift
//  ultralytics_yolo
//
//  Created by Sergio Sánchez on 9/11/23.
//

import AVFoundation
import Flutter
import Foundation

public class MethodCallHandler: NSObject, VideoCaptureDelegate, InferenceTimeListener,
  ResultsListener, FpsRateListener
{
  private let resultStreamHandler: ResultStreamHandler
  private let inferenceTimeStreamHandler: TimeStreamHandler
  private let fpsRateStreamHandler: TimeStreamHandler
  private var predictor: Predictor?
  private let videoCapture: VideoCapture

  public init(binaryMessenger: FlutterBinaryMessenger, videoCapture: VideoCapture) {
    self.videoCapture = videoCapture

    // Initialize stream handlers before super.init()
    self.resultStreamHandler = ResultStreamHandler()
    self.inferenceTimeStreamHandler = TimeStreamHandler()
    self.fpsRateStreamHandler = TimeStreamHandler()

    super.init()

    // Set up event channels after super.init()
    let resultsEventChannel = FlutterEventChannel(
      name: "ultralytics_yolo_prediction_results",
      binaryMessenger: binaryMessenger
    )
    resultsEventChannel.setStreamHandler(resultStreamHandler)

    let inferenceTimeEventChannel = FlutterEventChannel(
      name: "ultralytics_yolo_inference_time",
      binaryMessenger: binaryMessenger
    )
    inferenceTimeEventChannel.setStreamHandler(inferenceTimeStreamHandler)

    let fpsRateEventChannel = FlutterEventChannel(
      name: "ultralytics_yolo_fps_rate",
      binaryMessenger: binaryMessenger
    )
    fpsRateEventChannel.setStreamHandler(fpsRateStreamHandler)

    // Set up video capture delegate
    videoCapture.delegate = self
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    let args: [String: Any] = (call.arguments as? [String: Any]) ?? [:]

    switch call.method {
    case "loadModel":
      Task {
        await loadModel(args: args, result: result)
      }
    case "setConfidenceThreshold":
      setConfidenceThreshold(args: args, result: result)
    case "setIouThreshold":
      setIouThreshold(args: args, result: result)
    case "setNumItemsThreshold":
      setNumItemsThreshold(args: args, result: result)
    case "setLensDirection":
      setLensDirection(args: args, result: result)
    case "closeCamera":
      closeCamera(args: args, result: result)
    case "detectImage", "classifyImage":
      predictOnImage(args: args, result: result)
    default:
      result(FlutterMethodNotImplemented)
    }
  }

  // MARK: - VideoCaptureDelegate
  public func videoCapture(
    _ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer
  ) {
    predictor?.predict(
      sampleBuffer: sampleBuffer,
      onResultsListener: self,
      onInferenceTime: self,
      onFpsRate: self
    )
  }

  // MARK: - Model Loading and Configuration
  private func loadModel(args: [String: Any], result: @escaping FlutterResult) async {
    let flutterError = FlutterError(
      code: "PredictorError",
      message: "Invalid model",
      details: nil
    )

    guard let model = args["model"] as? [String: Any],
      let type = model["type"] as? String,
      let task = model["task"] as? String
    else {
      result(flutterError)
      return
    }

    var yoloModel: (any YoloModel)?

    switch type {
    case "local":
      guard let modelPath = model["modelPath"] as? String else {
        result(flutterError)
        return
      }
      yoloModel = LocalModel(modelPath: modelPath, task: task)
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
      case "classify":
        predictor = try await ObjectClassifier(yoloModel: yoloModel!)
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
    guard let conf = args["confidence"] as? Double else { return }
    (predictor as? ObjectDetector)?.setConfidenceThreshold(confidence: conf)
    result(nil)
  }

  private func setIouThreshold(args: [String: Any], result: @escaping FlutterResult) {
    guard let iou = args["iou"] as? Double else { return }
    (predictor as? ObjectDetector)?.setIouThreshold(iou: iou)
    result(nil)
  }

  private func setNumItemsThreshold(args: [String: Any], result: @escaping FlutterResult) {
    guard let numItems = args["numItems"] as? Int else { return }
    (predictor as? ObjectDetector)?.setNumItemsThreshold(numItems: numItems)
    result(nil)
  }

  private func setLensDirection(args: [String: Any], result: @escaping FlutterResult) {
    print("DEBUG: setLensDirection called with args:", args)
    guard let direction = args["direction"] as? Int else {
      print("DEBUG: Error - Invalid direction argument")
      result(
        FlutterError(code: "INVALID_ARGS", message: "Invalid direction argument", details: nil))
      return
    }

    guard let nativeView = self.videoCapture.nativeView else {
      print("DEBUG: Error - No nativeView found")
      result(
        FlutterError(
          code: "SWITCH_ERROR", message: "Failed to switch camera - no view", details: nil))
      return
    }

    // Execute camera switch on main thread
    DispatchQueue.main.async {
      print("DEBUG: Switching camera to direction:", direction)
      nativeView.switchCamera { success in
        if success {
          result("Success")
        } else {
          result(FlutterError(code: "SWITCH_ERROR", message: "Camera switch failed", details: nil))
        }
      }
    }
  }

  private func closeCamera(args: [String: Any], result: @escaping FlutterResult) {
    videoCapture.stop()
    result(nil)
  }

  private func createCIImage(fromPath path: String) throws -> CIImage? {
    let url = URL(fileURLWithPath: path)
    let data = try Data(contentsOf: url)
    return CIImage(data: data)
  }

  private func predictOnImage(args: [String: Any], result: @escaping FlutterResult) {
    guard let imagePath = args["imagePath"] as? String,
      let image = try? createCIImage(fromPath: imagePath)
    else {
      result(FlutterError(code: "PREDICT_ERROR", message: "Invalid image path", details: nil))
      return
    }

    predictor?.predictOnImage(image: image) { recognitions in
      result(recognitions)
    }
  }

  // MARK: - Listener Methods
  public func on(predictions: [[String: Any]]) {
    resultStreamHandler.sink(objects: predictions)
  }

  public func on(inferenceTime: Double) {
    inferenceTimeStreamHandler.sink(time: inferenceTime)
  }

  public func on(fpsRate: Double) {
    fpsRateStreamHandler.sink(time: fpsRate)
  }
}
