// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import AVFoundation
import Flutter
import UIKit

// Helper extension for Float to Double conversion
extension Float {
  var double: Double {
    return Double(self)
  }
}

@MainActor
public class SwiftYOLOPlatformView: NSObject, FlutterPlatformView, FlutterStreamHandler {
  private let frame: CGRect
  private let viewId: Int64
  private let messenger: FlutterBinaryMessenger
  private let flutterViewId: String

  // Event channel for sending detection results
  private let eventChannel: FlutterEventChannel
  private var eventSink: FlutterEventSink?

  // Method channel for receiving control commands
  private let methodChannel: FlutterMethodChannel

  // Reference to YOLOView
  private var yoloView: YOLOView?

  // Track current threshold values to maintain state
  private var currentConfidenceThreshold: Double = 0.25
  private var currentIouThreshold: Double = 0.7
  private var currentNumItemsThreshold: Int = 30

  init(
    frame: CGRect,
    viewId: Int64,
    args: Any?,
    messenger: FlutterBinaryMessenger
  ) {
    self.frame = frame
    self.viewId = viewId
    self.messenger = messenger

    // Get viewId passed from Flutter (primarily a string ID)
    if let dict = args as? [String: Any], let viewIdStr = dict["viewId"] as? String {
      self.flutterViewId = viewIdStr
    } else {
      // Fallback: Convert numeric viewId to string
      self.flutterViewId = "\(viewId)"
    }

    // Setup event channel - create unique channel name using view ID
    let eventChannelName = "com.ultralytics.yolo/detectionResults_\(flutterViewId)"
    self.eventChannel = FlutterEventChannel(name: eventChannelName, binaryMessenger: messenger)

    // Setup method channel - create unique channel name using view ID
    let methodChannelName = "com.ultralytics.yolo/controlChannel_\(flutterViewId)"
    self.methodChannel = FlutterMethodChannel(name: methodChannelName, binaryMessenger: messenger)

    super.init()

    // Set self as stream handler for event channel
    self.eventChannel.setStreamHandler(self)

    if let dict = args as? [String: Any],
      let modelName = dict["modelPath"] as? String,
      let taskRaw = dict["task"] as? String
    {
      let task = YOLOTask.fromString(taskRaw)

      // Get new threshold parameters
      let confidenceThreshold = dict["confidenceThreshold"] as? Double ?? 0.25
      let iouThreshold = dict["iouThreshold"] as? Double ?? 0.7
      let numItemsThreshold = dict["numItemsThreshold"] as? Int ?? 30
      let useGpu = dict["useGpu"] as? Bool ?? true

      // Get lensFacing parameter
      let lensFacingParam = dict["lensFacing"] as? String ?? "back"
      let cameraPosition: AVCaptureDevice.Position =
        (lensFacingParam.lowercased() == "front") ? .front : .back

      // Store initial thresholds
      self.currentConfidenceThreshold = confidenceThreshold
      self.currentIouThreshold = iouThreshold
      self.currentNumItemsThreshold = numItemsThreshold

      // Create YOLOView
      yoloView = YOLOView(
        frame: frame,
        modelPathOrName: modelName,
        task: task,
        useGpu: useGpu,
        cameraPosition: cameraPosition
      )

      // Configure YOLOView streaming functionality
      setupYOLOViewStreaming(args: dict)

      // Configure YOLOView
      updateThresholds(confidenceThreshold: confidenceThreshold, iouThreshold: iouThreshold)

      // Setup method channel handler
      setupMethodChannel()

      // Setup zoom callback — keep the legacy method-channel invocation for
      // existing consumers and also push a typed event on the event channel
      // so the new Dart-side ZoomIndicator (PR 3) can subscribe.
      yoloView?.onZoomChanged = { [weak self] zoomLevel in
        guard let self = self else { return }
        self.methodChannel.invokeMethod("onZoomChanged", arguments: Double(zoomLevel))
        self.sendStreamDataToFlutter([
          "type": "zoom",
          "value": Double(zoomLevel),
        ])
      }

      // Lens-change callback — emitted by YOLOView either when setLens is
      // called or when zoom crosses a lens boundary.
      yoloView?.onLensChanged = { [weak self] label in
        self?.sendStreamDataToFlutter([
          "type": "lens",
          "label": label,
        ])
      }

      // Tap-to-focus callback — fired after the native focus/exposure
      // configuration succeeds so the Dart FocusReticle can animate.
      yoloView?.onFocusTapped = { [weak self] x, y in
        self?.sendStreamDataToFlutter([
          "type": "focus",
          "x": Double(x),
          "y": Double(y),
        ])
      }

      // Register this view with the factory
      if let yoloView = yoloView {
        SwiftYOLOPlatformViewFactory.register(yoloView, for: Int(viewId))
      }
    }
  }

  // Overloaded method for setting just numItemsThreshold
  private func updateThresholds(numItemsThreshold: Int) {
    updateThresholds(
      confidenceThreshold: self.currentConfidenceThreshold,
      iouThreshold: nil,
      numItemsThreshold: numItemsThreshold
    )
  }

  // Method to update multiple thresholds
  private func updateThresholds(
    confidenceThreshold: Double, iouThreshold: Double?, numItemsThreshold: Int? = nil
  ) {
    guard let yoloView = yoloView else { return }

    // Update stored values
    self.currentConfidenceThreshold = confidenceThreshold
    if let iou = iouThreshold {
      self.currentIouThreshold = iou
    }
    if let numItems = numItemsThreshold {
      self.currentNumItemsThreshold = numItems
    }

    // Set confidence threshold
    yoloView.sliderConf.value = Float(confidenceThreshold)
    yoloView.sliderChanged(yoloView.sliderConf)

    // Set IoU threshold only if specified
    if let iou = iouThreshold {
      yoloView.sliderIoU.value = Float(iou)
      yoloView.sliderChanged(yoloView.sliderIoU)
    }

    // Set numItems threshold only if specified
    if let numItems = numItemsThreshold {
      yoloView.sliderNumItems.value = Float(numItems)
      yoloView.sliderChanged(yoloView.sliderNumItems)
    }
  }

  // Setup method channel call handler
  private func setupMethodChannel() {
    // Set method channel handler
    methodChannel.setMethodCallHandler { [weak self] (call, result) in
      guard let self = self else {
        result(
          FlutterError(
            code: "not_available", message: "YoloPlatformView was disposed", details: nil))
        return
      }

      switch call.method {
      case "setThreshold":
        // Maintained for backward compatibility
        if let args = call.arguments as? [String: Any],
          let threshold = args["threshold"] as? Double
        {
          self.updateThresholds(confidenceThreshold: threshold, iouThreshold: nil)
          result(nil)  // Success
        } else {
          result(
            FlutterError(
              code: "invalid_args", message: "Invalid arguments for setThreshold", details: nil))
        }

      case "setConfidenceThreshold":
        // Individual method for setting confidence threshold
        if let args = call.arguments as? [String: Any],
          let threshold = args["threshold"] as? Double
        {
          self.updateThresholds(
            confidenceThreshold: threshold,
            iouThreshold: nil,
            numItemsThreshold: nil
          )
          result(nil)  // Success
        } else {
          result(
            FlutterError(
              code: "invalid_args", message: "Invalid arguments for setConfidenceThreshold",
              details: nil))
        }

      case "setIoUThreshold", "setIouThreshold":
        // Individual method for setting IoU threshold
        if let args = call.arguments as? [String: Any],
          let threshold = args["threshold"] as? Double
        {
          self.updateThresholds(
            confidenceThreshold: self.currentConfidenceThreshold,
            iouThreshold: threshold,
            numItemsThreshold: nil
          )
          result(nil)  // Success
        } else {
          result(
            FlutterError(
              code: "invalid_args", message: "Invalid arguments for setIoUThreshold", details: nil))
        }

      case "setNumItemsThreshold":
        // New method for setting numItems threshold
        if let args = call.arguments as? [String: Any],
          let numItems = args["numItems"] as? Int
        {

          // Keep current confidence and IoU thresholds
          self.updateThresholds(
            numItemsThreshold: numItems
          )
          result(nil)  // Success
        } else {
          result(
            FlutterError(
              code: "invalid_args", message: "Invalid arguments for setNumItemsThreshold",
              details: nil))
        }

      case "setThresholds":
        // New method for setting multiple thresholds
        if let args = call.arguments as? [String: Any],
          let confidenceThreshold = args["confidenceThreshold"] as? Double
        {
          // IoU and numItems thresholds are optional
          let iouThreshold = args["iouThreshold"] as? Double
          let numItemsThreshold = args["numItemsThreshold"] as? Int

          self.updateThresholds(
            confidenceThreshold: confidenceThreshold,
            iouThreshold: iouThreshold,
            numItemsThreshold: numItemsThreshold
          )
          result(nil)  // Success
        } else {
          result(
            FlutterError(
              code: "invalid_args", message: "Invalid arguments for setThresholds", details: nil))
        }

      case "switchCamera":

        self.yoloView?.switchCameraTapped()
        result(nil)  // Success

      case "setTorchMode":
        if let args = call.arguments as? [String: Any],
          let enabled = args["enabled"] as? Bool
        {
          self.yoloView?.setTorchMode(enabled)
          result(nil)  // Success
        } else {
          result(
            FlutterError(
              code: "invalid_args", message: "Invalid arguments for setTorchMode", details: nil))
        }

      case "setZoomLevel":
        if let args = call.arguments as? [String: Any],
          let zoomLevel = args["zoomLevel"] as? Double
        {

          self.yoloView?.setZoomLevel(CGFloat(zoomLevel))
          result(nil)  // Success
        } else {
          result(
            FlutterError(
              code: "invalid_args", message: "Invalid arguments for setZoomLevel", details: nil))
        }

      case "setStreamingConfig":
        // Method to update streaming configuration
        if let args = call.arguments as? [String: Any] {

          let streamConfig = YOLOStreamConfig.from(dict: args)
          self.yoloView?.setStreamConfig(streamConfig)

          result(nil)  // Success
        } else {
          result(
            FlutterError(
              code: "invalid_args", message: "Invalid arguments for setStreamingConfig",
              details: nil
            ))
        }

      case "stop":
        // Method to stop camera and inference
        self.stopCamera()
        result(nil)  // Success

      case "pause":
        // Captures the next frame into the cached share image, then stops
        // the session — matches upstream YOLO iOS pauseTapped.
        if let view = self.yoloView {
          view.pause { _ in result(nil) }
        } else {
          result(nil)
        }

      case "resume":
        // Resumes from a pause()-induced state, clearing the cached frame.
        self.yoloView?.resume()
        result(nil)

      case "setModel":
        // Method to dynamically switch models
        if let args = call.arguments as? [String: Any],
          let modelPath = args["modelPath"] as? String,
          let taskString = args["task"] as? String
        {
          let task = YOLOTask.fromString(taskString)
          let useGpu = args["useGpu"] as? Bool ?? true

          // Use YOLOView's setModel method to switch the model
          self.yoloView?.setModel(modelPathOrName: modelPath, task: task, useGpu: useGpu) {
            modelResult in
            switch modelResult {
            case .success:

              result(nil)  // Success
            case .failure(let error):

              result(
                FlutterError(
                  code: "MODEL_NOT_FOUND",
                  message: "Failed to load model: \(modelPath) - \(error.localizedDescription)",
                  details: nil
                )
              )
            }
          }
        } else {
          result(
            FlutterError(
              code: "invalid_args", message: "Invalid arguments for setModel", details: nil))
        }

      case "captureFrame":
        // Legacy alias — always returns the composited share image.
        self.yoloView?.capturePhoto(withOverlays: true) { image in
          if let image, let data = image.jpegData(compressionQuality: 0.9) {
            result(FlutterStandardTypedData(bytes: data))
          } else {
            result(
              FlutterError(
                code: "capture_failed",
                message: "Failed to capture photo from camera",
                details: nil))
          }
        }

      case "capturePhoto":
        // Canonical capture endpoint. Honors `withOverlays` (default true):
        // false returns the raw oriented camera frame for callers that want
        // to do their own annotation. Behavior matches the Android handler.
        let withOverlays = (call.arguments as? [String: Any])?["withOverlays"] as? Bool ?? true
        self.yoloView?.capturePhoto(withOverlays: withOverlays) { image in
          if let image, let data = image.jpegData(compressionQuality: 0.9) {
            result(FlutterStandardTypedData(bytes: data))
          } else {
            result(
              FlutterError(
                code: "capture_failed",
                message: "Failed to capture photo from camera",
                details: nil))
          }
        }

      case "getAvailableLenses":
        // Enumerate physical lenses for the current camera position.
        guard let yoloView = self.yoloView else {
          result([] as [Any])
          return
        }
        let lenses = yoloView.availableLenses().map { lens -> [String: Any] in
          return [
            "zoomFactor": Double(lens.zoomFactor),
            "label": lens.label,
          ]
        }
        result(lenses)

      case "setLens":
        // Switch to the lens whose zoom factor most closely matches the
        // requested value, then emit a `lens` event.
        if let args = call.arguments as? [String: Any],
          let zoomFactor = args["zoomFactor"] as? Double
        {
          self.yoloView?.setLens(zoomFactor: CGFloat(zoomFactor))
          result(nil)
        } else {
          result(
            FlutterError(
              code: "invalid_args", message: "Invalid arguments for setLens", details: nil))
        }

      case "tapToFocus":
        // x, y are normalized 0..1 view-relative coordinates; native sets
        // focusPointOfInterest + exposurePointOfInterest and emits a
        // `focus` event so the Dart FocusReticle can animate.
        if let args = call.arguments as? [String: Any],
          let x = args["x"] as? Double,
          let y = args["y"] as? Double
        {
          self.yoloView?.tapToFocus(x: CGFloat(x), y: CGFloat(y))
          result(nil)
        } else {
          result(
            FlutterError(
              code: "invalid_args", message: "Invalid arguments for tapToFocus", details: nil))
        }

      // Additional methods can be added here in the future

      default:
        result(FlutterMethodNotImplemented)
      }
    }
  }

  /// Configure YOLOView streaming functionality based on creation parameters
  private func setupYOLOViewStreaming(args: [String: Any]) {
    guard let yoloView = yoloView else { return }

    // Parse streaming configuration from args
    let streamingConfigParam = args["streamingConfig"] as? [String: Any]

    let streamConfig: YOLOStreamConfig
    if let configDict = streamingConfigParam {

      streamConfig = YOLOStreamConfig.from(dict: configDict)
    } else {
      // Use default minimal configuration for optimal performance

      streamConfig = YOLOStreamConfig.DEFAULT
    }

    // Use the parsed stream config (no more hardcoding)
    let finalStreamConfig = streamConfig

    // Configure YOLOView with the stream config
    yoloView.setStreamConfig(finalStreamConfig)

    // Set up streaming callback to forward data to Flutter via event channel
    yoloView.setStreamCallback { [weak self] streamData in
      self?.sendStreamDataToFlutter(streamData)
    }
  }

  /// Send stream data to Flutter via event channel
  private func sendStreamDataToFlutter(_ streamData: [String: Any]) {
    guard let eventSink = self.eventSink else { return }
    DispatchQueue.main.async {
      eventSink(streamData)
    }
  }

  public func view() -> UIView {
    return yoloView ?? UIView()
  }

  // MARK: - FlutterStreamHandler Protocol

  public func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink)
    -> FlutterError?
  {
    self.eventSink = events
    return nil
  }

  public func onCancel(withArguments arguments: Any?) -> FlutterError? {
    self.eventSink = nil
    return nil
  }

  // MARK: - Cleanup

  /// Stop camera and inference operations
  private func stopCamera() {

    // Stop the camera capture
    yoloView?.stop()

    // Clear callbacks to prevent retain cycles
    yoloView?.onDetection = nil
    yoloView?.onZoomChanged = nil
    yoloView?.onLensChanged = nil
    yoloView?.onFocusTapped = nil
    yoloView?.setStreamCallback(nil)

    // Remove from factory registry
    SwiftYOLOPlatformViewFactory.unregister(for: Int(viewId))

  }

  deinit {
    // Clean up event channel
    eventSink = nil
    eventChannel.setStreamHandler(nil)

    // Clean up method channel
    methodChannel.setMethodCallHandler(nil)

    let yoloViewToClean = yoloView
    let viewIdToUnregister = Int(viewId)
    let instanceIdToRemove = flutterViewId

    yoloView = nil

    Task { @MainActor in
      // Stop the camera capture
      yoloViewToClean?.stop()

      // Clear callbacks to prevent retain cycles
      yoloViewToClean?.onDetection = nil
      yoloViewToClean?.onZoomChanged = nil
      yoloViewToClean?.onLensChanged = nil
      yoloViewToClean?.onFocusTapped = nil
      yoloViewToClean?.setStreamCallback(nil)

      // Remove from factory registry
      SwiftYOLOPlatformViewFactory.unregister(for: viewIdToUnregister)

      // Dispose model instance from YOLOInstanceManager
      YOLOInstanceManager.shared.removeInstance(instanceId: instanceIdToRemove)
    }
  }
}
