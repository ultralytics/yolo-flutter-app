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
    let flutterViewId: String
    if let dict = args as? [String: Any], let viewIdStr = dict["viewId"] as? String {
      flutterViewId = viewIdStr
      print("SwiftYOLOPlatformView: Using Flutter-provided viewId: \(flutterViewId)")
    } else {
      // Fallback: Convert numeric viewId to string
      flutterViewId = "\(viewId)"
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

      // Configure YOLOView
      setupYOLOView(confidenceThreshold: confidenceThreshold, iouThreshold: iouThreshold)

      // Setup method channel handler
      setupMethodChannel()
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

    // Setup detection result callback
    yoloView.onDetection = { [weak self] result in
      print(
        "SwiftYOLOPlatformView: onDetection callback triggered with \(result.boxes.count) detections"
      )

      guard let self = self else {
        print("SwiftYOLOPlatformView: self is nil in onDetection callback")
        return
      }

      guard let eventSink = self.eventSink else {
        print("SwiftYOLOPlatformView: eventSink is nil - no listener for events")
        return
      }

      // Convert detection results to Flutter-compatible map
      let resultMap = self.convertYOLOResultToMap(result)
      if let detections = resultMap["detections"] as? [[String: Any]] {
        print("SwiftYOLOPlatformView: Converted result to map with \(detections.count) detections")
      } else {
        print("SwiftYOLOPlatformView: Converted result but no valid detections found")
      }

      // Send event on main thread
      DispatchQueue.main.async {
        print("SwiftYOLOPlatformView: Sending event to Flutter via eventSink")
        eventSink(resultMap)
      }
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

      // Additional methods can be added here in the future

      default:
        result(FlutterMethodNotImplemented)
      }
    }
  }

  // Helper method to convert YOLOResult to Flutter-compatible map
  private func convertYOLOResultToMap(_ result: YOLOResult) -> [String: Any] {
    var resultMap: [String: Any] = [:]

    // Convert detection results
    var detections: [[String: Any]] = []
    for box in result.boxes {
      var detection: [String: Any] = [
        "classIndex": box.index,
        "className": box.cls,
        "confidence": Double(box.conf),
        "boundingBox": [
          "left": box.xywh.origin.x,
          "top": box.xywh.origin.y,
          "right": box.xywh.origin.x + box.xywh.size.width,
          "bottom": box.xywh.origin.y + box.xywh.size.height,
        ],
        "normalizedBox": [
          "left": box.xywhn.origin.x,
          "top": box.xywhn.origin.y,
          "right": box.xywhn.origin.x + box.xywhn.size.width,
          "bottom": box.xywhn.origin.y + box.xywhn.size.height,
        ],
      ]

      // Extended data for pose detection or segmentation can be added as needed
      // Currently only basic detection information is sent

      detections.append(detection)
    }

    resultMap["detections"] = detections
    resultMap["processingTimeMs"] = result.speed * 1000  // Convert seconds to milliseconds
    if let fpsValue = result.fps {
      resultMap["fps"] = fpsValue
    }

    // Add annotated image if available (optional)
    if let annotatedImage = result.annotatedImage,
      let imageData = annotatedImage.jpegData(compressionQuality: 0.9)
    {
      resultMap["annotatedImage"] = FlutterStandardTypedData(bytes: imageData)
    }

    return resultMap
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

  deinit {
    // Clean up event channel
    eventSink = nil
    eventChannel.setStreamHandler(nil)

    // Clean up method channel
    methodChannel.setMethodCallHandler(nil)

    // Clean up YOLOView
    // Only set to nil because MainActor-isolated methods can't be called directly
    yoloView = nil

    // Note: stop() method call was removed due to MainActor issues
    // If setting up later in a Task, use code like this:
    // Task { @MainActor in
    //    self.yoloView?.stop()
    // }
  }
}
