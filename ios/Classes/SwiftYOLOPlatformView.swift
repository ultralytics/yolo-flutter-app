// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

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
      print("SwiftYOLOPlatformView: Using Flutter-provided viewId: \(flutterViewId)")
    } else {
      // Fallback: Convert numeric viewId to string
      self.flutterViewId = "\(viewId)"
      print("SwiftYOLOPlatformView: Using fallback numeric viewId: \(flutterViewId)")
    }

    // Setup event channel - create unique channel name using view ID
    let eventChannelName = "com.ultralytics.yolo/detectionResults_\(flutterViewId)"
    print("SwiftYOLOPlatformView: Creating event channel with name: \(eventChannelName)")
    self.eventChannel = FlutterEventChannel(name: eventChannelName, binaryMessenger: messenger)

    // Setup method channel - create unique channel name using view ID
    let methodChannelName = "com.ultralytics.yolo/controlChannel_\(flutterViewId)"
    print("SwiftYOLOPlatformView: Creating method channel with name: \(methodChannelName)")
    self.methodChannel = FlutterMethodChannel(name: methodChannelName, binaryMessenger: messenger)

    super.init()

    // Set self as stream handler for event channel
    self.eventChannel.setStreamHandler(self)

    // Unwrap creation parameters
    if let dict = args as? [String: Any],
      let modelName = dict["modelPath"] as? String,
      let taskRaw = dict["task"] as? String
    {
      let task = YOLOTask.fromString(taskRaw)

      print("SwiftYOLOPlatformView: Received modelPath: \(modelName)")

      // Get new threshold parameters
      let confidenceThreshold = dict["confidenceThreshold"] as? Double ?? 0.5
      let iouThreshold = dict["iouThreshold"] as? Double ?? 0.45

      // Old threshold parameter for backward compatibility
      let oldThreshold = dict["threshold"] as? Double ?? 0.5

      // Determine which thresholds to use (prioritize new parameters)
      print(
        "SwiftYOLOPlatformView: Received thresholds - confidence: \(confidenceThreshold), IoU: \(iouThreshold), old: \(oldThreshold)"
      )

      // Create YOLOView
      yoloView = YOLOView(
        frame: frame,
        modelPathOrName: modelName,
        task: task
      )

      // Hide native UI controls by default
      yoloView?.showUIControls = false

      // Configure YOLOView streaming functionality
      setupYOLOViewStreaming(args: dict)

      // Configure YOLOView
      setupYOLOView(confidenceThreshold: confidenceThreshold, iouThreshold: iouThreshold)

      // Setup method channel handler
      setupMethodChannel()

      // Setup zoom callback
      yoloView?.onZoomChanged = { [weak self] zoomLevel in
        self?.methodChannel.invokeMethod("onZoomChanged", arguments: Double(zoomLevel))
      }

      // Register this view with the factory
      if let yoloView = yoloView {
        SwiftYOLOPlatformViewFactory.register(yoloView, for: Int(viewId))
      }
    }
  }

  // Method for backward compatibility
  private func setupYOLOView(threshold: Double) {
    setupYOLOView(confidenceThreshold: threshold, iouThreshold: 0.45)  // Use default IoU value
  }

  // Setup YOLOView and connect callbacks (using new parameters)
  private func setupYOLOView(confidenceThreshold: Double, iouThreshold: Double) {
    guard let yoloView = yoloView else { return }

    // Debug information
    print(
      "SwiftYOLOPlatformView: setupYOLOView - Setting up detection callback with confidenceThreshold: \(confidenceThreshold), iouThreshold: \(iouThreshold)"
    )

    // YOLOView streaming is now configured separately
    // Keep simple detection callback for compatibility
    yoloView.onDetection = { result in
      print(
        "SwiftYOLOPlatformView: onDetection callback triggered with \(result.boxes.count) detections"
      )
    }

    // Set thresholds
    updateThresholds(confidenceThreshold: confidenceThreshold, iouThreshold: iouThreshold)
  }

  // Method to update threshold (kept for backward compatibility)
  private func updateThreshold(threshold: Double) {
    updateThresholds(confidenceThreshold: threshold, iouThreshold: nil)
  }

  // Overloaded method for setting just numItemsThreshold
  private func updateThresholds(numItemsThreshold: Int) {
    updateThresholds(
      confidenceThreshold: Double(self.yoloView?.sliderConf.value ?? 0.5),
      iouThreshold: nil,
      numItemsThreshold: numItemsThreshold
    )
  }

  // Method to update multiple thresholds
  private func updateThresholds(
    confidenceThreshold: Double, iouThreshold: Double?, numItemsThreshold: Int? = nil
  ) {
    guard let yoloView = yoloView else { return }

    print(
      "SwiftYoloPlatformView: Updating thresholds - confidence: \(confidenceThreshold), IoU: \(String(describing: iouThreshold)), numItems: \(String(describing: numItemsThreshold))"
    )

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
          print("SwiftYOLOPlatformView: Received setThreshold call with threshold: \(threshold)")
          self.updateThreshold(threshold: threshold)
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
          print(
            "SwiftYoloPlatformView: Received setConfidenceThreshold call with value: \(threshold)")
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
          print("SwiftYOLOPlatformView: Received setIoUThreshold call with value: \(threshold)")
          self.updateThresholds(
            confidenceThreshold: Double(self.yoloView?.sliderConf.value ?? 0.5),
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
          print("SwiftYOLOPlatformView: Received setNumItemsThreshold call with value: \(numItems)")
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

          print(
            "SwiftYoloPlatformView: Received setThresholds call with confidence: \(confidenceThreshold), IoU: \(String(describing: iouThreshold)), numItems: \(String(describing: numItemsThreshold))"
          )
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

      case "setShowUIControls":
        // Method to toggle native UI controls visibility
        if let args = call.arguments as? [String: Any],
          let show = args["show"] as? Bool
        {
          print("SwiftYOLOPlatformView: Setting UI controls visibility to \(show)")
          yoloView?.showUIControls = show
          result(nil)  // Success
        } else {
          result(
            FlutterError(
              code: "invalid_args", message: "Invalid arguments for setShowUIControls", details: nil
            ))
        }

      case "switchCamera":
        print("SwiftYoloPlatformView: Received switchCamera call")
        self.yoloView?.switchCameraTapped()
        result(nil)  // Success

      case "setZoomLevel":
        if let args = call.arguments as? [String: Any],
          let zoomLevel = args["zoomLevel"] as? Double
        {
          print("SwiftYoloPlatformView: Received setZoomLevel call with value: \(zoomLevel)")
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
          print("SwiftYOLOPlatformView: Received setStreamingConfig call")
          let streamConfig = YOLOStreamConfig.from(dict: args)
          self.yoloView?.setStreamConfig(streamConfig)
          print("SwiftYOLOPlatformView: YOLOView streaming config updated")
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
        print("SwiftYOLOPlatformView: Received stop call from Flutter")
        self.stopCamera()
        result(nil)  // Success

      case "setModel":
        // Method to dynamically switch models
        if let args = call.arguments as? [String: Any],
          let modelPath = args["modelPath"] as? String,
          let taskString = args["task"] as? String
        {
          let task = YOLOTask.fromString(taskString)
          print(
            "SwiftYOLOPlatformView: Received setModel call with modelPath: \(modelPath), task: \(taskString)"
          )

          // Use YOLOView's setModel method to switch the model
          self.yoloView?.setModel(modelPathOrName: modelPath, task: task) { modelResult in
            switch modelResult {
            case .success:
              print("SwiftYOLOPlatformView: Model switched successfully")
              result(nil)  // Success
            case .failure(let error):
              print("SwiftYOLOPlatformView: Failed to switch model: \(error.localizedDescription)")
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
        // Method to capture current camera frame with detection overlays
        print("SwiftYOLOPlatformView: Received captureFrame call")
        self.yoloView?.capturePhoto { [weak self] image in
          if let image = image {
            // Convert UIImage to byte array (JPEG format)
            if let imageData = image.jpegData(compressionQuality: 0.9) {
              // Convert to FlutterStandardTypedData for efficient transfer
              let flutterData = FlutterStandardTypedData(bytes: imageData)
              result(flutterData)
            } else {
              result(
                FlutterError(
                  code: "conversion_failed",
                  message: "Failed to convert captured image to JPEG data",
                  details: nil
                )
              )
            }
          } else {
            result(
              FlutterError(
                code: "capture_failed",
                message: "Failed to capture photo from camera",
                details: nil
              )
            )
          }
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
      print("SwiftYOLOPlatformView: Creating YOLOStreamConfig from creation params: \(configDict)")
      streamConfig = YOLOStreamConfig.from(dict: configDict)
    } else {
      // Use default minimal configuration for optimal performance
      print("SwiftYOLOPlatformView: Using default streaming config")
      streamConfig = YOLOStreamConfig.DEFAULT
    }

    // Configure YOLOView with the stream config
    yoloView.setStreamConfig(streamConfig)
    print("SwiftYOLOPlatformView: YOLOView streaming configured: \(streamConfig)")

    // Set up streaming callback to forward data to Flutter via event channel
    yoloView.setStreamCallback { [weak self] streamData in
      // Forward streaming data from YOLOView to Flutter
      self?.sendStreamDataToFlutter(streamData)
    }
  }

  /// Send stream data to Flutter via event channel
  private func sendStreamDataToFlutter(_ streamData: [String: Any]) {
    print(
      "SwiftYOLOPlatformView: Sending stream data to Flutter: \(streamData.keys.joined(separator: ", "))"
    )

    guard let eventSink = self.eventSink else {
      print("SwiftYOLOPlatformView: eventSink is nil - no listener for events")
      return
    }

    // Send event on main thread
    DispatchQueue.main.async {
      print("SwiftYOLOPlatformView: Sending stream data to Flutter via eventSink")
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
    print("SwiftYOLOPlatformView: onListen called - Stream handler connected")
    self.eventSink = events
    print("SwiftYOLOPlatformView: eventSink set successfully")
    return nil
  }

  public func onCancel(withArguments arguments: Any?) -> FlutterError? {
    print("SwiftYOLOPlatformView: onCancel called - Stream handler disconnected")
    self.eventSink = nil
    return nil
  }

  // MARK: - Cleanup

  /// Stop camera and inference operations
  private func stopCamera() {
    print("SwiftYOLOPlatformView: Stopping camera and inference")

    // Stop the camera capture
    yoloView?.stop()

    // Clear callbacks to prevent retain cycles
    yoloView?.onDetection = nil
    yoloView?.onZoomChanged = nil
    yoloView?.setStreamCallback(nil)

    // Remove from factory registry
    SwiftYOLOPlatformViewFactory.unregister(for: Int(viewId))

    print("SwiftYOLOPlatformView: Camera stopped successfully")
  }

  deinit {
    print(
      "SwiftYOLOPlatformView: deinit called for viewId: \(viewId), flutterViewId: \(flutterViewId)")

    // Dispose model instance from YOLOInstanceManager
    // Since we're in deinit and YOLOInstanceManager is @MainActor, we need to dispatch
    let instanceIdToRemove = flutterViewId
    print(
      "SwiftYOLOPlatformView: Scheduling disposal of model instance with id: \(instanceIdToRemove)")

    Task { @MainActor in
      YOLOInstanceManager.shared.removeInstance(instanceId: instanceIdToRemove)
      print("SwiftYOLOPlatformView: Model instance disposed: \(instanceIdToRemove)")
    }

    // Clean up event channel
    eventSink = nil
    eventChannel.setStreamHandler(nil)

    // Clean up method channel
    methodChannel.setMethodCallHandler(nil)

    // Clean up YOLOView reference - its own deinit will handle camera cleanup
    yoloView = nil

    print("SwiftYOLOPlatformView: deinit completed - cleanup scheduled")
  }
}
