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

    // Track current threshold values to maintain state
    private var currentConfidenceThreshold: Double = 0.5
    private var currentIouThreshold: Double = 0.45
    private var currentNumItemsThreshold: Int = 30
    private var currentShowOverlays: Bool = true

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
        self.methodChannel = FlutterMethodChannel(
            name: methodChannelName, binaryMessenger: messenger)

        super.init()

        // Set self as stream handler for event channel
        self.eventChannel.setStreamHandler(self)

        // Unwrap creation parameters
        if let dict = args as? [String: Any] {
        }

        if let dict = args as? [String: Any] {
            // Prefer 'models' list when provided (temporary until multi-model implemented)
            var modelName: String? = dict["modelPath"] as? String
            var taskRaw: String? = dict["task"] as? String
            if let models = dict["models"] as? [[String: Any]], let first = models.first {
                if let mp = first["modelPath"] as? String { modelName = mp }
                if let t = first["task"] as? String { taskRaw = t }
            }
            guard let modelName = modelName, let taskRaw = taskRaw else { return }
            let task = YOLOTask.fromString(taskRaw)

            // Get new threshold parameters
            let confidenceThreshold = dict["confidenceThreshold"] as? Double ?? 0.5
            let iouThreshold = dict["iouThreshold"] as? Double ?? 0.45
            let numItemsThreshold = dict["numItemsThreshold"] as? Int ?? 30
            let showOverlays = dict["showOverlays"] as? Bool ?? true

            // Store initial thresholds
            self.currentConfidenceThreshold = confidenceThreshold
            self.currentIouThreshold = iouThreshold
            self.currentNumItemsThreshold = numItemsThreshold
            self.currentShowOverlays = showOverlays

            // Old threshold parameter for backward compatibility
            let oldThreshold = dict["threshold"] as? Double ?? 0.5

            // Determine which thresholds to use (prioritize new parameters)

            // Create YOLOView
            yoloView = YOLOView(
                frame: frame,
                modelPathOrName: modelName,
                task: task
            )

            // Hide native UI controls by default
            yoloView?.showUIControls = false

            // Set overlay visibility
            yoloView?.showOverlays = showOverlays

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
                // If full models list provided, prefer initializing via setModels
                if let models = dict["models"] as? [[String: Any]], !models.isEmpty {
                    let pairs: [(String, YOLOTask)] = models.compactMap { m in
                        guard let path = m["modelPath"] as? String,
                            let taskStr = m["task"] as? String
                        else { return nil }
                        return (path, YOLOTask.fromString(taskStr))
                    }
                    if !pairs.isEmpty {
                        yoloView.setModels(pairs) { _ in /* ignore init completion */ }
                    }
                }
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

        // YOLOView streaming is now configured separately
        // Keep simple detection callback for compatibility
        yoloView.onDetection = { result in
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
                        code: "not_available", message: "YoloPlatformView was disposed",
                        details: nil))
                return
            }

            switch call.method {
            case "setThreshold":
                // Maintained for backward compatibility
                if let args = call.arguments as? [String: Any],
                    let threshold = args["threshold"] as? Double
                {
                    self.updateThreshold(threshold: threshold)
                    result(nil)  // Success
                } else {
                    result(
                        FlutterError(
                            code: "invalid_args", message: "Invalid arguments for setThreshold",
                            details: nil))
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
                            code: "invalid_args",
                            message: "Invalid arguments for setConfidenceThreshold",
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
                            code: "invalid_args", message: "Invalid arguments for setIoUThreshold",
                            details: nil))
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
                            code: "invalid_args",
                            message: "Invalid arguments for setNumItemsThreshold",
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
                            code: "invalid_args", message: "Invalid arguments for setThresholds",
                            details: nil))
                }

            case "setShowUIControls":
                // Method to toggle native UI controls visibility
                if let args = call.arguments as? [String: Any],
                    let show = args["show"] as? Bool
                {

                    yoloView?.showUIControls = show
                    result(nil)  // Success
                } else {
                    result(
                        FlutterError(
                            code: "invalid_args",
                            message: "Invalid arguments for setShowUIControls", details: nil
                        ))
                }

            case "setShowOverlays":
                // Method to toggle bounding box overlay visibility
                if let args = call.arguments as? [String: Any],
                    let show = args["show"] as? Bool
                {
                    self.currentShowOverlays = show
                    yoloView?.showOverlays = show
                    result(nil)  // Success
                } else {
                    result(
                        FlutterError(
                            code: "invalid_args", message: "Invalid arguments for setShowOverlays",
                            details: nil
                        ))
                }

            case "switchCamera":

                self.yoloView?.switchCameraTapped()
                result(nil)  // Success

            case "setZoomLevel":
                if let args = call.arguments as? [String: Any],
                    let zoomLevel = args["zoomLevel"] as? Double
                {

                    self.yoloView?.setZoomLevel(CGFloat(zoomLevel))
                    result(nil)  // Success
                } else {
                    result(
                        FlutterError(
                            code: "invalid_args", message: "Invalid arguments for setZoomLevel",
                            details: nil))
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
                            code: "invalid_args",
                            message: "Invalid arguments for setStreamingConfig",
                            details: nil
                        ))
                }

            case "stop":
                // Method to stop camera and inference

                self.stopCamera()
                result(nil)  // Success

            case "setModel":
                // Method to dynamically switch models
                if let args = call.arguments as? [String: Any],
                    let modelPath = args["modelPath"] as? String,
                    let taskString = args["task"] as? String
                {
                    let task = YOLOTask.fromString(taskString)

                    // Use YOLOView's setModel method to switch the model
                    self.yoloView?.setModel(modelPathOrName: modelPath, task: task) { modelResult in
                        switch modelResult {
                        case .success:

                            result(nil)  // Success
                        case .failure(let error):

                            result(
                                FlutterError(
                                    code: "MODEL_NOT_FOUND",
                                    message:
                                        "Failed to load model: \(modelPath) - \(error.localizedDescription)",
                                    details: nil
                                )
                            )
                        }
                    }
                } else {
                    result(
                        FlutterError(
                            code: "invalid_args", message: "Invalid arguments for setModel",
                            details: nil))
                }

            case "setModels":
                // Multi-model setter: parse full models list and forward to YOLOView.setModels
                if let args = call.arguments as? [String: Any],
                    let models = args["models"] as? [[String: Any]],
                    !models.isEmpty
                {
                    let pairs: [(String, YOLOTask)] = models.compactMap { m in
                        guard let path = m["modelPath"] as? String,
                            let taskStr = m["task"] as? String
                        else { return nil }
                        return (path, YOLOTask.fromString(taskStr))
                    }
                    if pairs.isEmpty {
                        result(
                            FlutterError(
                                code: "invalid_args",
                                message: "No valid models provided",
                                details: nil
                            )
                        )
                    } else {
                        self.yoloView?.setModels(pairs) { setResult in
                            switch setResult {
                            case .success:
                                result(nil)
                            case .failure(let error):
                                result(
                                    FlutterError(
                                        code: "MODEL_NOT_FOUND",
                                        message:
                                            "Failed to load models: \(error.localizedDescription)",
                                        details: nil
                                    )
                                )
                            }
                        }
                    }
                } else {
                    result(
                        FlutterError(
                            code: "invalid_args",
                            message: "models is required and must be non-empty",
                            details: nil
                        )
                    )
                }

            case "captureFrame":
                // Method to capture current camera frame with detection overlays

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

            case "setModels":
                // Method to set multiple models (initial support: apply the first model)
                if let args = call.arguments as? [String: Any],
                    let models = args["models"] as? [[String: Any]],
                    let first = models.first,
                    let modelPath = first["modelPath"] as? String,
                    let taskString = first["task"] as? String
                {
                    let task = YOLOTask.fromString(taskString)
                    self.yoloView?.setModel(modelPathOrName: modelPath, task: task) { modelResult in
                        switch modelResult {
                        case .success:
                            result(nil)  // Success
                        case .failure(let error):
                            result(
                                FlutterError(
                                    code: "MODEL_NOT_FOUND",
                                    message:
                                        "Failed to load model: \(modelPath) - \(error.localizedDescription)",
                                    details: nil
                                )
                            )
                        }
                    }
                } else {
                    result(
                        FlutterError(
                            code: "invalid_args", message: "Invalid arguments for setModels",
                            details: nil))
                }

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

    public func onListen(
        withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink
    )
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
            yoloViewToClean?.setStreamCallback(nil)

            // Remove from factory registry
            SwiftYOLOPlatformViewFactory.unregister(for: viewIdToUnregister)

            // Dispose model instance from YOLOInstanceManager
            YOLOInstanceManager.shared.removeInstance(instanceId: instanceIdToRemove)
        }
    }
}
