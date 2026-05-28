// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Ultralytics YOLO Package, providing the core UI component for real-time object detection.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLOView class is the primary UI component for displaying real-time YOLO model results.
//  It handles camera setup, model loading, video frame processing, rendering of detection results,
//  and user interactions such as pinch-to-zoom. The view can display bounding boxes, masks for segmentation,
//  pose estimation keypoints, and oriented bounding boxes depending on the active task. It includes
//  UI elements for controlling inference settings such as confidence threshold and IoU threshold,
//  and provides functionality for capturing photos with detection results overlaid.

import AVFoundation
import UIKit
import Vision

/// A UIView component that provides real-time object detection, segmentation, and pose estimation capabilities.
@MainActor
public class YOLOView: UIView, VideoCaptureDelegate {
  func onInferenceTime(speed: Double, fps: Double) {
    // Store performance data for streaming
    self.currentFps = fps
    self.currentProcessingTime = speed

    if showUIControls {
      DispatchQueue.main.async {
        self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", fps, speed)  // t2 seconds to ms
      }
    }
  }

  func onPredict(result: YOLOResult) {

    // Check if we should process inference result based on frequency control
    if !shouldRunInference() {
      return
    }

    task == .obb ? showOBBs(predictions: result) : showBoxes(predictions: result)
    onDetection?(result)

    // Streaming callback (with output throttling)
    if let streamCallback = onStream {
      if shouldProcessFrame() {
        updateLastInferenceTime()

        // Convert to stream data and send
        let streamData = convertResultToStreamData(result)
        // Add timestamp and frame info
        var enhancedStreamData = streamData
        enhancedStreamData["timestamp"] = Int64(Date().timeIntervalSince1970 * 1000)  // milliseconds
        enhancedStreamData["frameNumber"] = frameNumberCounter
        frameNumberCounter += 1

        streamCallback(enhancedStreamData)
      }
    }

    if task == .segment || task == .semantic {
      DispatchQueue.main.async {
        let maskImage =
          self.task == .segment ? result.masks?.combinedMask : result.semanticMask?.maskImage
        if let maskImage {

          guard let maskLayer = self.maskLayer else { return }

          maskLayer.isHidden = false

          maskLayer.frame = self.overlayLayer.bounds
          maskLayer.contents = maskImage

          self.videoCapture.predictor.isUpdating = false
        } else {
          self.videoCapture.predictor.isUpdating = false
        }
      }
    } else if task == .classify {
      self.overlayYOLOClassificationsCALayer(on: self, result: result)
    } else if task == .pose {
      self.removeAllSubLayers(parentLayer: poseLayer)
      var keypointList = [[(x: Float, y: Float)]]()
      var confsList = [[Float]]()

      for keypoint in result.keypointsList {
        keypointList.append(keypoint.xyn)
        confsList.append(keypoint.conf)
      }
      guard let poseLayer = poseLayer else { return }
      drawKeypoints(
        keypointsList: keypointList, confsList: confsList, boundingBoxes: result.boxes,
        on: poseLayer, imageViewSize: overlayLayer.frame.size, originalImageSize: result.orig_shape)
    }
  }

  var onDetection: ((YOLOResult) -> Void)?

  // Streaming functionality
  private var streamConfig: YOLOStreamConfig?
  var onStream: (([String: Any]) -> Void)?

  // Frame counter for streaming
  private var frameNumberCounter: Int64 = 0

  // Throttling variables for performance control
  private var lastInferenceTime: TimeInterval = 0
  private var targetFrameInterval: TimeInterval? = nil  // in seconds
  private var throttleInterval: TimeInterval? = nil  // in seconds

  // Inference frequency control variables
  private var inferenceFrameInterval: TimeInterval? = nil  // Target inference interval in seconds
  private var frameSkipCount: Int = 0  // Current frame skip counter
  private var targetSkipFrames: Int = 0  // Number of frames to skip between inferences

  // Performance data tracking
  private var currentFps: Double = 0.0
  private var currentProcessingTime: Double = 0.0

  private var videoCapture: VideoCapture
  private var busy = false
  private var currentBuffer: CVPixelBuffer?
  var framesDone = 0
  var task = YOLOTask.detect
  var colors: [String: UIColor] = [:]
  var modelName: String = ""
  var classes: [String] = []
  let maxBoundingBoxViews = 100
  var boundingBoxViews = [BoundingBoxView]()
  public var sliderNumItems = UISlider()
  public var labelSliderNumItems = UILabel()
  public var sliderConf = UISlider()
  public var labelSliderConf = UILabel()
  public var sliderIoU = UISlider()
  public var labelSliderIoU = UILabel()
  public var labelName = UILabel()
  public var labelFPS = UILabel()
  public var labelZoom = UILabel()
  public var activityIndicator = UIActivityIndicatorView()
  public var playButton = UIButton()
  public var pauseButton = UIButton()
  public var switchCameraButton = UIButton()
  public var toolbar = UIView()
  let selection = UISelectionFeedbackGenerator()
  private var overlayLayer = CALayer()
  private var maskLayer: CALayer?
  private var poseLayer: CALayer?

  // Flag to control UI visibility (sliders, buttons, etc.)
  private var _showUIControls: Bool = false

  /// Property to get or set the visibility of UI controls
  public var showUIControls: Bool {
    get { return _showUIControls }
    set {
      _showUIControls = newValue
      updateUIControlsVisibility()
    }
  }

  // Flag to control bounding box overlay visibility
  private var _showOverlays: Bool = true

  /// Property to get or set the visibility of bounding box overlays
  public var showOverlays: Bool {
    get { return _showOverlays }
    set {
      _showOverlays = newValue
    }
  }

  private let minimumZoom: CGFloat = 1.0
  private let maximumZoom: CGFloat = 10.0
  private var lastZoomFactor: CGFloat = 1.0
  /// Cached frame captured at pause so `capturePhoto` after `pause()` returns
  /// the paused frame instead of asking a stopped session for a new buffer.
  /// Matches upstream YOLO iOS pause/share semantics.
  private var pausedShareImage: UIImage?

  /// Callback for zoom level changes
  public var onZoomChanged: ((CGFloat) -> Void)?

  /// Callback for lens changes (emitted when the active lens label changes
  /// either via `setLens` or because zoom crossed a lens boundary).
  public var onLensChanged: ((String) -> Void)?

  /// Callback for tap-to-focus events (normalized 0..1 view-relative coords).
  public var onFocusTapped: ((CGFloat, CGFloat) -> Void)?

  public var capturedImage: UIImage?

  // Lens-snap state (ported from yolo-ios-app YOLOView.swift:1157-1185).
  private let physicalLensTypes: [AVCaptureDevice.DeviceType] = [
    .builtInUltraWideCamera,
    .builtInWideAngleCamera,
    .builtInTelephotoCamera,
  ]
  private var currentLensLabel: String = ""

  // Camera-flip blur transition (ported from yolo-ios-app YOLOView.swift:1036-1060).
  private weak var cameraTransitionView: UIView?

  public init(
    frame: CGRect,
    modelPathOrName: String,
    task: YOLOTask,
    useGpu: Bool = true,
    cameraPosition: AVCaptureDevice.Position = .back
  ) {
    self.videoCapture = VideoCapture()
    super.init(frame: frame)
    setModel(modelPathOrName: modelPathOrName, task: task, useGpu: useGpu)
    setUpOrientationChangeNotification()
    self.setUpBoundingBoxViews()
    self.setupUI()
    self.videoCapture.delegate = self
    // Hide UI controls by default
    self.showUIControls = false
    start(position: cameraPosition)
    setupOverlayLayer()
  }

  required init?(coder: NSCoder) {
    self.videoCapture = VideoCapture()
    super.init(coder: coder)
  }

  public override func awakeFromNib() {
    super.awakeFromNib()
    Task { @MainActor in
      setUpOrientationChangeNotification()
      setUpBoundingBoxViews()
      setupUI()
      videoCapture.delegate = self
      // Hide UI controls by default
      self.showUIControls = false
      start(position: .back)
      setupOverlayLayer()
    }
  }

  public func setModel(
    modelPathOrName: String,
    task: YOLOTask,
    useGpu: Bool = true,
    completion: ((Result<Void, Error>) -> Void)? = nil
  ) {
    activityIndicator.startAnimating()
    boundingBoxViews.forEach { box in
      box.hide()
    }
    removeClassificationLayers()

    self.task = task
    setupSublayers()

    var modelURL: URL?
    let lowercasedPath = modelPathOrName.lowercased()
    let fileManager = FileManager.default

    // Determine model URL
    if lowercasedPath.hasSuffix(".mlmodel") || lowercasedPath.hasSuffix(".mlpackage")
      || lowercasedPath.hasSuffix(".mlmodelc")
    {
      let possibleURL = URL(fileURLWithPath: modelPathOrName)
      var isDirectory: ObjCBool = false
      if fileManager.fileExists(atPath: possibleURL.path, isDirectory: &isDirectory) {
        modelURL = possibleURL
      }
    } else {
      if let compiledURL = Bundle.main.url(forResource: modelPathOrName, withExtension: "mlmodelc")
      {
        modelURL = compiledURL
      } else if let packageURL = Bundle.main.url(
        forResource: modelPathOrName, withExtension: "mlpackage")
      {
        modelURL = packageURL
      }
    }

    guard let unwrappedModelURL = modelURL else {
      // Model not found - allow camera preview without inference
      NSLog(
        "YOLOView: Model file not found: %@. Camera will run without inference.", modelPathOrName)
      self.videoCapture.predictor = nil
      self.activityIndicator.stopAnimating()
      self.labelName.text = "No Model"
      // Call completion with success to allow camera to start
      completion?(.success(()))
      return
    }

    modelName = unwrappedModelURL.deletingPathExtension().lastPathComponent

    // Common success handling for all tasks
    func handleSuccess(predictor: Predictor) {
      // Release old predictor before setting new one to prevent memory leak
      if self.videoCapture.predictor != nil {
        self.videoCapture.predictor = nil
      }

      self.videoCapture.predictor = predictor

      // Set stream configuration for original image capture
      if let basePredictor = predictor as? BasePredictor {
        basePredictor.streamConfig = self.streamConfig
      }

      self.activityIndicator.stopAnimating()
      self.labelName.text = modelName
      completion?(.success(()))
    }

    // Common failure handling for all tasks
    func handleFailure(_ error: Error) {
      NSLog("YOLOView: Failed to load model: %@", String(describing: error))
      self.activityIndicator.stopAnimating()
      completion?(.failure(error))
    }

    switch task {
    case .classify:
      Classifier.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true, useGpu: useGpu) {
        [weak self] result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .segment:
      Segmenter.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true, useGpu: useGpu) {
        [weak self] result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .semantic:
      SemanticSegmenter.create(
        unwrappedModelURL: unwrappedModelURL, isRealTime: true, useGpu: useGpu
      ) {
        [weak self] result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .pose:
      PoseEstimater.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true, useGpu: useGpu) {
        [weak self] result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .obb:
      ObbDetector.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true, useGpu: useGpu) {
        [weak self] result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    default:
      ObjectDetector.create(unwrappedModelURL: unwrappedModelURL, isRealTime: true, useGpu: useGpu)
      {
        [weak self] result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }
    }
  }

  private func start(position: AVCaptureDevice.Position) {
    if !busy {
      busy = true
      videoCapture.setUp(
        sessionPreset: .photo, position: position, videoOrientation: currentVideoOrientation()
      ) {
        success in
        // .hd4K3840x2160 or .photo (4032x3024)  Warning: 4k may not work on all devices i.e. 2019 iPod
        if success {
          // Add the video preview into the UI.
          if let previewLayer = self.videoCapture.previewLayer {
            self.layer.insertSublayer(previewLayer, at: 0)
            self.videoCapture.previewLayer?.frame = self.bounds  // resize preview layer
            for box in self.boundingBoxViews {
              box.addToLayer(previewLayer)
            }
          }
          self.videoCapture.previewLayer?.addSublayer(self.overlayLayer)
          // Once everything is set up, we can start capturing live video.
          self.videoCapture.start()
        } else {
          NSLog(
            "YOLOView: Failed to set up camera - permission may be denied or camera unavailable")
        }
        self.busy = false
      }
    }
  }

  public func setCameraPosition(_ position: AVCaptureDevice.Position) {

    let savedDelegate = videoCapture.delegate
    let savedPredictor = videoCapture.predictor

    videoCapture.stop()

    videoCapture.delegate = savedDelegate
    videoCapture.predictor = savedPredictor

    start(position: position)
  }

  public func stop() {
    videoCapture.stop()
    videoCapture.delegate = nil
    // Release predictor to prevent memory leak
    videoCapture.predictor = nil
  }

  /// Pause the camera session, first snapshotting the next frame into
  /// `pausedShareImage` so `capturePhoto` can return it without re-running
  /// the session. Mirrors upstream YOLO iOS `pauseTapped`.
  public func pause(completion: ((Void) -> Void)? = nil) {
    videoCapture.captureNextFrame { [weak self] image in
      self?.pausedShareImage = image
      self?.videoCapture.stop()
      completion?(())
    }
  }

  /// Resume after `pause()`; clears the cached share frame and restarts the
  /// session. Use this instead of `restartCamera()` when the session was
  /// paused via `pause()`.
  public func resume() {
    pausedShareImage = nil
    videoCapture.start()
  }

  func setUpBoundingBoxViews() {
    // Ensure all bounding box views are initialized up to the maximum allowed.
    while boundingBoxViews.count < maxBoundingBoxViews {
      boundingBoxViews.append(BoundingBoxView())
    }

  }

  func setupOverlayLayer() {
    let width = self.bounds.width
    let height = self.bounds.height

    var ratio: CGFloat = 1.0
    if videoCapture.captureSession.sessionPreset == .photo {
      ratio = (4.0 / 3.0)
    } else {
      ratio = (16.0 / 9.0)
    }
    var offSet = CGFloat.zero
    var margin = CGFloat.zero
    if self.bounds.width < self.bounds.height {
      offSet = height / ratio
      margin = (offSet - self.bounds.width) / 2
      self.overlayLayer.frame = CGRect(
        x: -margin, y: 0, width: offSet, height: self.bounds.height)
    } else {
      offSet = width / ratio
      margin = (offSet - self.bounds.height) / 2
      self.overlayLayer.frame = CGRect(
        x: 0, y: -margin, width: self.bounds.width, height: offSet)
    }

    // Update mask layer frame to match overlay layer bounds
    if let maskLayer = self.maskLayer {
      maskLayer.frame = self.overlayLayer.bounds
    }
  }

  func setupMaskLayerIfNeeded() {
    if maskLayer == nil {
      let layer = CALayer()
      layer.frame = self.overlayLayer.bounds
      layer.opacity = 0.5
      layer.name = "maskLayer"
      layer.magnificationFilter = .linear
      layer.minificationFilter = .linear

      self.overlayLayer.addSublayer(layer)
      self.maskLayer = layer
    }
  }

  func setupPoseLayerIfNeeded() {
    if poseLayer == nil {
      let layer = CALayer()
      layer.frame = self.overlayLayer.bounds
      layer.opacity = 0.5
      self.overlayLayer.addSublayer(layer)
      self.poseLayer = layer
    }
  }

  public func resetLayers() {
    removeAllSubLayers(parentLayer: maskLayer)
    removeAllSubLayers(parentLayer: poseLayer)
    removeAllSubLayers(parentLayer: overlayLayer)

    maskLayer = nil
    poseLayer = nil
  }

  func setupSublayers() {
    resetLayers()

    switch task {
    case .segment, .semantic:
      setupMaskLayerIfNeeded()
    case .pose:
      setupPoseLayerIfNeeded()
    default: break
    }
  }

  func removeAllSubLayers(parentLayer: CALayer?) {
    guard let parentLayer = parentLayer else { return }
    parentLayer.sublayers?.forEach { layer in
      layer.removeFromSuperlayer()
    }
    parentLayer.sublayers = nil
    parentLayer.contents = nil
  }

  func addMaskSubLayers() {
    guard let maskLayer = maskLayer else { return }
    self.overlayLayer.addSublayer(maskLayer)
  }

  func showBoxes(predictions: YOLOResult) {

    let width = self.bounds.width
    let height = self.bounds.height
    var resultCount = 0

    resultCount = predictions.boxes.count

    if UIDevice.current.orientation == .portrait {

      var ratio: CGFloat = 1.0

      if videoCapture.captureSession.sessionPreset == .photo {
        ratio = (height / width) / (4.0 / 3.0)
      } else {
        ratio = (height / width) / (16.0 / 9.0)
      }

      if showUIControls {
        self.labelSliderNumItems.text =
          String(resultCount) + " items (max " + String(Int(sliderNumItems.value)) + ")"
      }
      for i in 0..<boundingBoxViews.count {
        if i < (resultCount) && i < 50 {
          var rect = CGRect.zero
          var label = ""
          var boxColor: UIColor = .white
          var confidence: CGFloat = 0
          var alpha: CGFloat = 0.9
          var bestClass = ""

          switch task {
          case .detect:
            let prediction = predictions.boxes[i]
            rect = CGRect(
              x: prediction.xywhn.minX, y: 1 - prediction.xywhn.maxY, width: prediction.xywhn.width,
              height: prediction.xywhn.height)
            bestClass = prediction.cls
            confidence = CGFloat(prediction.conf)
            let colorIndex = prediction.index % ultralyticsColors.count
            boxColor = ultralyticsColors[colorIndex]
            label = DetectionLabelStyle.text(className: bestClass, confidence: confidence)
            alpha = DetectionLabelStyle.alpha(confidence: confidence)
          default:
            let prediction = predictions.boxes[i]
            rect = prediction.xywhn
            bestClass = prediction.cls
            confidence = CGFloat(prediction.conf)
            label = DetectionLabelStyle.text(className: bestClass, confidence: confidence)
            let colorIndex = prediction.index % ultralyticsColors.count
            boxColor = ultralyticsColors[colorIndex]
            alpha = DetectionLabelStyle.alpha(confidence: confidence)

          }
          var displayRect = rect
          switch UIDevice.current.orientation {
          case .portraitUpsideDown:
            displayRect = CGRect(
              x: 1.0 - rect.origin.x - rect.width,
              y: 1.0 - rect.origin.y - rect.height,
              width: rect.width,
              height: rect.height)
          case .landscapeLeft:
            displayRect = CGRect(
              x: rect.origin.x,
              y: rect.origin.y,
              width: rect.width,
              height: rect.height)
          case .landscapeRight:
            displayRect = CGRect(
              x: rect.origin.x,
              y: rect.origin.y,
              width: rect.width,
              height: rect.height)
          case .unknown:
            fallthrough
          default: break
          }
          if ratio >= 1 {
            let offset = (1 - ratio) * (0.5 - displayRect.minX)
            if task == .detect {
              let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
              displayRect = displayRect.applying(transform)
            } else {
              let transform = CGAffineTransform(translationX: offset, y: 0)
              displayRect = displayRect.applying(transform)
            }
            displayRect.size.width *= ratio
          } else {
            if task == .detect {
              let offset = (ratio - 1) * (0.5 - displayRect.maxY)

              let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
              displayRect = displayRect.applying(transform)
            } else {
              let offset = (ratio - 1) * (0.5 - displayRect.minY)
              let transform = CGAffineTransform(translationX: 0, y: offset)
              displayRect = displayRect.applying(transform)
            }
            ratio = (height / width) / (3.0 / 4.0)
            displayRect.size.height /= ratio
          }
          displayRect = VNImageRectForNormalizedRect(displayRect, Int(width), Int(height))

          if _showOverlays {
            boundingBoxViews[i].show(
              frame: displayRect, label: label, color: boxColor, alpha: alpha)
          } else {
            boundingBoxViews[i].hide()
          }

        } else {
          boundingBoxViews[i].hide()
        }
      }
    } else {
      resultCount = predictions.boxes.count
      if showUIControls {
        self.labelSliderNumItems.text =
          String(resultCount) + " items (max " + String(Int(sliderNumItems.value)) + ")"
      }

      let frameAspectRatio = videoCapture.longSide / videoCapture.shortSide
      let viewAspectRatio = width / height
      var scaleX: CGFloat = 1.0
      var scaleY: CGFloat = 1.0
      var offsetX: CGFloat = 0.0
      var offsetY: CGFloat = 0.0

      if frameAspectRatio > viewAspectRatio {
        scaleY = height / videoCapture.shortSide
        scaleX = scaleY
        offsetX = (videoCapture.longSide * scaleX - width) / 2
      } else {
        scaleX = width / videoCapture.longSide
        scaleY = scaleX
        offsetY = (videoCapture.shortSide * scaleY - height) / 2
      }

      for i in 0..<boundingBoxViews.count {
        if i < resultCount && i < 50 {
          var rect = CGRect.zero
          var label = ""
          var boxColor: UIColor = .white
          var confidence: CGFloat = 0
          var alpha: CGFloat = 0.9
          var bestClass = ""

          switch task {
          case .detect:
            let prediction = predictions.boxes[i]
            // For the detect task, invert y using "1 - maxY" as before
            rect = CGRect(
              x: prediction.xywhn.minX,
              y: 1 - prediction.xywhn.maxY,
              width: prediction.xywhn.width,
              height: prediction.xywhn.height
            )
            bestClass = prediction.cls
            confidence = CGFloat(prediction.conf)

          default:
            let prediction = predictions.boxes[i]
            rect = CGRect(
              x: prediction.xywhn.minX,
              y: 1 - prediction.xywhn.maxY,
              width: prediction.xywhn.width,
              height: prediction.xywhn.height
            )
            bestClass = prediction.cls
            confidence = CGFloat(prediction.conf)
          }

          let colorIndex = predictions.boxes[i].index % ultralyticsColors.count
          boxColor = ultralyticsColors[colorIndex]
          label = DetectionLabelStyle.text(className: bestClass, confidence: confidence)
          alpha = DetectionLabelStyle.alpha(confidence: confidence)

          rect.origin.x = rect.origin.x * videoCapture.longSide * scaleX - offsetX
          rect.origin.y =
            height
            - (rect.origin.y * videoCapture.shortSide * scaleY
              - offsetY
              + rect.size.height * videoCapture.shortSide * scaleY)
          rect.size.width *= videoCapture.longSide * scaleX
          rect.size.height *= videoCapture.shortSide * scaleY

          if _showOverlays {
            boundingBoxViews[i].show(
              frame: rect,
              label: label,
              color: boxColor,
              alpha: alpha
            )
          } else {
            boundingBoxViews[i].hide()
          }
        } else {
          boundingBoxViews[i].hide()
        }
      }
    }
  }

  func showOBBs(predictions: YOLOResult) {
    let resultCount = predictions.obb.count
    if showUIControls {
      self.labelSliderNumItems.text =
        String(resultCount) + " items (max " + String(Int(sliderNumItems.value)) + ")"
    }

    let overlayFrame = overlayLayer.frame
    for i in 0..<boundingBoxViews.count {
      guard i < resultCount && i < 50 else {
        boundingBoxViews[i].hide()
        continue
      }

      let detection = predictions.obb[i]
      let box = detection.box
      let confidence = CGFloat(detection.confidence)
      let rect = CGRect(
        x: overlayFrame.minX + CGFloat(box.cx - box.w / 2) * overlayFrame.width,
        y: overlayFrame.minY + CGFloat(box.cy - box.h / 2) * overlayFrame.height,
        width: CGFloat(box.w) * overlayFrame.width,
        height: CGFloat(box.h) * overlayFrame.height
      )
      if _showOverlays {
        boundingBoxViews[i].show(
          frame: rect,
          label: DetectionLabelStyle.text(className: detection.cls, confidence: confidence),
          color: ultralyticsColors[detection.index % ultralyticsColors.count],
          alpha: DetectionLabelStyle.alpha(confidence: confidence),
          angle: CGFloat(box.angle)
        )
      } else {
        boundingBoxViews[i].hide()
      }
    }
  }

  func removeClassificationLayers() {
    if let sublayers = self.layer.sublayers {
      for layer in sublayers where layer.name == "YOLOOverlayLayer" {
        layer.removeFromSuperlayer()
      }
    }
  }

  func overlayYOLOClassificationsCALayer(on view: UIView, result: YOLOResult) {

    removeClassificationLayers()

    let overlayLayer = CALayer()
    overlayLayer.frame = view.bounds
    overlayLayer.name = "YOLOOverlayLayer"

    guard let top1 = result.probs?.top1Label,
      let top1Conf = result.probs?.top1Conf
    else {
      return
    }

    var colorIndex = 0
    if let index = result.names.firstIndex(of: top1) {
      colorIndex = index % ultralyticsColors.count
    }
    let color = ultralyticsColors[colorIndex]

    let confidence = CGFloat(top1Conf)
    let labelText = DetectionLabelStyle.text(className: top1, confidence: confidence)

    let textLayer = CATextLayer()
    let fontSize: CGFloat = 18
    DetectionLabelStyle.configure(textLayer, fontSize: fontSize)
    textLayer.string = labelText
    let alpha = DetectionLabelStyle.alpha(confidence: confidence)
    textLayer.foregroundColor = UIColor.white.withAlphaComponent(alpha).cgColor
    textLayer.backgroundColor = color.withAlphaComponent(alpha).cgColor
    let textSize = DetectionLabelStyle.size(for: labelText, fontSize: fontSize)
    textLayer.frame = CGRect(
      x: (bounds.width - textSize.width) / 2,
      y: (bounds.height - textSize.height) / 2,
      width: textSize.width,
      height: textSize.height
    )

    overlayLayer.addSublayer(textLayer)

    view.layer.addSublayer(overlayLayer)
  }

  private func setupUI() {
    labelName.text = modelName
    labelName.textAlignment = .center
    labelName.font = UIFont.systemFont(ofSize: 24, weight: .medium)
    labelName.textColor = .black
    labelName.font = UIFont.preferredFont(forTextStyle: .title1)
    self.addSubview(labelName)

    labelFPS.text = String(format: "%.1f FPS - %.1f ms", 0.0, 0.0)
    labelFPS.textAlignment = .center
    labelFPS.textColor = .black
    labelFPS.font = UIFont.preferredFont(forTextStyle: .body)
    self.addSubview(labelFPS)

    labelSliderNumItems.text = "0 items (max 30)"
    labelSliderNumItems.textAlignment = .left
    labelSliderNumItems.textColor = .black
    labelSliderNumItems.font = UIFont.preferredFont(forTextStyle: .subheadline)
    self.addSubview(labelSliderNumItems)

    sliderNumItems.minimumValue = 0
    sliderNumItems.maximumValue = 100
    sliderNumItems.value = 30
    sliderNumItems.minimumTrackTintColor = .darkGray
    sliderNumItems.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)
    sliderNumItems.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
    self.addSubview(sliderNumItems)

    labelSliderConf.text = "0.25 Confidence Threshold"
    labelSliderConf.textAlignment = .left
    labelSliderConf.textColor = .black
    labelSliderConf.font = UIFont.preferredFont(forTextStyle: .subheadline)
    self.addSubview(labelSliderConf)

    sliderConf.minimumValue = 0
    sliderConf.maximumValue = 1
    sliderConf.value = 0.25
    sliderConf.minimumTrackTintColor = .darkGray
    sliderConf.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)
    sliderConf.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
    self.addSubview(sliderConf)

    labelSliderIoU.text = "0.7 IoU Threshold"
    labelSliderIoU.textAlignment = .left
    labelSliderIoU.textColor = .black
    labelSliderIoU.font = UIFont.preferredFont(forTextStyle: .subheadline)
    self.addSubview(labelSliderIoU)

    sliderIoU.minimumValue = 0
    sliderIoU.maximumValue = 1
    sliderIoU.value = 0.7
    sliderIoU.minimumTrackTintColor = .darkGray
    sliderIoU.maximumTrackTintColor = .systemGray.withAlphaComponent(0.7)
    sliderIoU.addTarget(self, action: #selector(sliderChanged), for: .valueChanged)
    self.addSubview(sliderIoU)

    if showUIControls {
      self.labelSliderNumItems.text = "0 items (max " + String(Int(sliderNumItems.value)) + ")"
    }
    self.labelSliderConf.text = "0.25 Confidence Threshold"
    self.labelSliderIoU.text = "0.7 IoU Threshold"

    labelZoom.text = "1.00x"
    labelZoom.textColor = .black
    labelZoom.font = UIFont.systemFont(ofSize: 14)
    labelZoom.textAlignment = .center
    labelZoom.font = UIFont.preferredFont(forTextStyle: .body)
    self.addSubview(labelZoom)

    let config = UIImage.SymbolConfiguration(pointSize: 20, weight: .regular, scale: .default)

    playButton.setImage(UIImage(systemName: "play.fill", withConfiguration: config), for: .normal)
    playButton.tintColor = .systemGray
    pauseButton.setImage(UIImage(systemName: "pause.fill", withConfiguration: config), for: .normal)
    pauseButton.tintColor = .systemGray
    switchCameraButton = UIButton()
    switchCameraButton.setImage(
      UIImage(systemName: "camera.rotate", withConfiguration: config), for: .normal)
    switchCameraButton.tintColor = .systemGray
    playButton.isEnabled = false
    pauseButton.isEnabled = true
    playButton.addTarget(self, action: #selector(playTapped), for: .touchUpInside)
    pauseButton.addTarget(self, action: #selector(pauseTapped), for: .touchUpInside)
    switchCameraButton.addTarget(self, action: #selector(switchCameraTapped), for: .touchUpInside)
    toolbar.backgroundColor = .darkGray.withAlphaComponent(0.7)
    self.addSubview(toolbar)
    toolbar.addSubview(playButton)
    toolbar.addSubview(pauseButton)
    toolbar.addSubview(switchCameraButton)
    // Dart owns gestures (pinch + tap) via Flutter GestureDetector in YOLOShowcase;
    // native is setter-only. Do not attach UIPinchGestureRecognizer here.
  }

  /// Update the visibility of UI controls based on the showUIControls flag
  private func updateUIControlsVisibility() {
    // Elements to hide/show
    let controlElements: [UIView] = [
      labelSliderNumItems, sliderNumItems,
      labelSliderConf, sliderConf,
      labelSliderIoU, sliderIoU,
      labelName, labelFPS, labelZoom,
      toolbar, playButton, pauseButton, switchCameraButton,
    ]

    // Set visibility for all UI elements
    for element in controlElements {
      element.isHidden = !_showUIControls
    }

    // Force layout update
    self.setNeedsLayout()
  }

  public override func layoutSubviews() {
    setupOverlayLayer()
    let isLandscape = bounds.width > bounds.height
    activityIndicator.frame = CGRect(x: center.x - 50, y: center.y - 50, width: 100, height: 100)
    if isLandscape {
      toolbar.backgroundColor = .clear
      playButton.tintColor = .darkGray
      pauseButton.tintColor = .darkGray
      switchCameraButton.tintColor = .darkGray

      let width = bounds.width
      let height = bounds.height

      let topMargin: CGFloat = 0

      let titleLabelHeight: CGFloat = height * 0.1
      labelName.frame = CGRect(
        x: 0,
        y: topMargin,
        width: width,
        height: titleLabelHeight
      )

      let subLabelHeight: CGFloat = height * 0.04
      labelFPS.frame = CGRect(
        x: 0,
        y: center.y - height * 0.24 - subLabelHeight,
        width: width,
        height: subLabelHeight
      )

      let sliderWidth: CGFloat = width * 0.2
      let sliderHeight: CGFloat = height * 0.1

      labelSliderNumItems.frame = CGRect(
        x: width * 0.1,
        y: labelFPS.frame.minY - sliderHeight,
        width: sliderWidth,
        height: sliderHeight
      )

      sliderNumItems.frame = CGRect(
        x: width * 0.1,
        y: labelSliderNumItems.frame.maxY + 10,
        width: sliderWidth,
        height: sliderHeight
      )

      labelSliderConf.frame = CGRect(
        x: width * 0.1,
        y: sliderNumItems.frame.maxY + 10,
        width: sliderWidth * 1.5,
        height: sliderHeight
      )

      sliderConf.frame = CGRect(
        x: width * 0.1,
        y: labelSliderConf.frame.maxY + 10,
        width: sliderWidth,
        height: sliderHeight
      )

      labelSliderIoU.frame = CGRect(
        x: width * 0.1,
        y: sliderConf.frame.maxY + 10,
        width: sliderWidth * 1.5,
        height: sliderHeight
      )

      sliderIoU.frame = CGRect(
        x: width * 0.1,
        y: labelSliderIoU.frame.maxY + 10,
        width: sliderWidth,
        height: sliderHeight
      )

      let zoomLabelWidth: CGFloat = width * 0.2
      labelZoom.frame = CGRect(
        x: center.x - zoomLabelWidth / 2,
        y: self.bounds.maxY - 120,
        width: zoomLabelWidth,
        height: height * 0.03
      )

      let toolBarHeight: CGFloat = 66
      let buttonHeihgt: CGFloat = toolBarHeight * 0.75
      toolbar.frame = CGRect(x: 0, y: height - toolBarHeight, width: width, height: toolBarHeight)
      playButton.frame = CGRect(x: 0, y: 0, width: buttonHeihgt, height: buttonHeihgt)
      pauseButton.frame = CGRect(
        x: playButton.frame.maxX, y: 0, width: buttonHeihgt, height: buttonHeihgt)
      switchCameraButton.frame = CGRect(
        x: pauseButton.frame.maxX, y: 0, width: buttonHeihgt, height: buttonHeihgt)
    } else {
      toolbar.backgroundColor = .darkGray.withAlphaComponent(0.7)
      playButton.tintColor = .systemGray
      pauseButton.tintColor = .systemGray
      switchCameraButton.tintColor = .systemGray

      let width = bounds.width
      let height = bounds.height

      let topMargin: CGFloat = height * 0.02

      let titleLabelHeight: CGFloat = height * 0.1
      labelName.frame = CGRect(
        x: 0,
        y: topMargin,
        width: width,
        height: titleLabelHeight
      )

      let subLabelHeight: CGFloat = height * 0.04
      labelFPS.frame = CGRect(
        x: 0,
        y: labelName.frame.maxY + 15,
        width: width,
        height: subLabelHeight
      )

      let sliderWidth: CGFloat = width * 0.46
      let sliderHeight: CGFloat = height * 0.02

      sliderNumItems.frame = CGRect(
        x: width * 0.01,
        y: center.y - sliderHeight - height * 0.24,
        width: sliderWidth,
        height: sliderHeight
      )

      labelSliderNumItems.frame = CGRect(
        x: width * 0.01,
        y: sliderNumItems.frame.minY - sliderHeight - 10,
        width: sliderWidth,
        height: sliderHeight
      )

      labelSliderConf.frame = CGRect(
        x: width * 0.01,
        y: center.y + height * 0.24,
        width: sliderWidth * 1.5,
        height: sliderHeight
      )

      sliderConf.frame = CGRect(
        x: width * 0.01,
        y: labelSliderConf.frame.maxY + 10,
        width: sliderWidth,
        height: sliderHeight
      )

      labelSliderIoU.frame = CGRect(
        x: width * 0.01,
        y: sliderConf.frame.maxY + 10,
        width: sliderWidth * 1.5,
        height: sliderHeight
      )

      sliderIoU.frame = CGRect(
        x: width * 0.01,
        y: labelSliderIoU.frame.maxY + 10,
        width: sliderWidth,
        height: sliderHeight
      )

      let zoomLabelWidth: CGFloat = width * 0.2
      labelZoom.frame = CGRect(
        x: center.x - zoomLabelWidth / 2,
        y: self.bounds.maxY - 120,
        width: zoomLabelWidth,
        height: height * 0.03
      )

      let toolBarHeight: CGFloat = 66
      let buttonHeihgt: CGFloat = toolBarHeight * 0.75
      toolbar.frame = CGRect(x: 0, y: height - toolBarHeight, width: width, height: toolBarHeight)
      playButton.frame = CGRect(x: 0, y: 0, width: buttonHeihgt, height: buttonHeihgt)
      pauseButton.frame = CGRect(
        x: playButton.frame.maxX, y: 0, width: buttonHeihgt, height: buttonHeihgt)
      switchCameraButton.frame = CGRect(
        x: pauseButton.frame.maxX, y: 0, width: buttonHeihgt, height: buttonHeihgt)
    }

    self.videoCapture.previewLayer?.frame = self.bounds
  }

  private func setUpOrientationChangeNotification() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(orientationDidChange),
      name: UIDevice.orientationDidChangeNotification, object: nil)
  }

  @objc func orientationDidChange() {
    videoCapture.updateVideoOrientation(orientation: currentVideoOrientation())
  }

  private func currentVideoOrientation() -> AVCaptureVideoOrientation {
    if let interfaceOrientation = window?.windowScene?.interfaceOrientation {
      switch interfaceOrientation {
      case .portrait:
        return .portrait
      case .portraitUpsideDown:
        return .portraitUpsideDown
      case .landscapeLeft:
        return .landscapeLeft
      case .landscapeRight:
        return .landscapeRight
      default:
        break
      }
    }

    switch UIDevice.current.orientation {
    case .portrait:
      return .portrait
    case .portraitUpsideDown:
      return .portraitUpsideDown
    case .landscapeLeft:
      return .landscapeRight
    case .landscapeRight:
      return .landscapeLeft
    default:
      return videoCapture.previewLayer?.connection?.videoOrientation ?? .portrait
    }
  }

  @objc func sliderChanged(_ sender: Any) {

    if sender as? UISlider === sliderNumItems {
      if let basePredictor = videoCapture.predictor as? BasePredictor {
        let numItems = Int(sliderNumItems.value)
        basePredictor.setNumItemsThreshold(numItems: numItems)
      }
    }
    let conf = Double(round(100 * sliderConf.value)) / 100
    let iou = Double(round(100 * sliderIoU.value)) / 100
    self.labelSliderConf.text = String(conf) + " Confidence Threshold"
    self.labelSliderIoU.text = String(iou) + " IoU Threshold"
    // Apply thresholds to all predictor types via BasePredictor
    if let basePredictor = videoCapture.predictor as? BasePredictor {
      basePredictor.setIouThreshold(iou: iou)
      basePredictor.setConfidenceThreshold(confidence: conf)
    }
  }

  /// Set the camera zoom level programmatically
  public func setZoomLevel(_ zoomLevel: CGFloat) {
    guard let device = videoCapture.captureDevice else { return }

    // Return zoom value between the minimum and maximum zoom values
    func minMaxZoom(_ factor: CGFloat) -> CGFloat {
      return min(min(max(factor, minimumZoom), maximumZoom), device.activeFormat.videoMaxZoomFactor)
    }

    let newZoomFactor = minMaxZoom(zoomLevel)

    do {
      try device.lockForConfiguration()
      defer {
        device.unlockForConfiguration()
      }
      device.videoZoomFactor = newZoomFactor
      lastZoomFactor = newZoomFactor

      // Update zoom label
      self.labelZoom.text = String(format: "%.1fx", newZoomFactor)

      // Notify zoom change
      onZoomChanged?(newZoomFactor)

      // Emit a lens-change event if the active lens (per the lens-snap math)
      // changed as a result of this zoom step.
      updateSelectedLensLabel(rawZoomFactor: newZoomFactor, device: device)
    } catch {
      NSLog("YOLOView: Failed to set zoom level: %@", error.localizedDescription)
    }
  }

  // MARK: - Multi-lens support
  //
  // Port of the lens enumeration + lens-snap math from
  // `yolo-ios-app/Sources/YOLO/YOLOView.swift:1157-1185` and the device
  // discovery in `VideoCapture.swift:32-45`. Setters only — Dart owns
  // gestures; this class never attaches a pinch/tap recognizer.

  /// Returns the physical lens devices available for the active camera
  /// position (back: ultra-wide / wide / telephoto, front: a single device).
  /// Each entry pairs the canonical user-facing label with the raw zoom
  /// factor on the currently active (virtual) device.
  public func availableLenses() -> [(zoomFactor: CGFloat, label: String)] {
    let position = videoCapture.captureDevice?.position ?? .back
    let activeDevice = videoCapture.captureDevice

    if position == .front {
      guard let device = activeDevice else { return [] }
      return [(1.0, lensLabel(for: device))]
    }

    let discovery = AVCaptureDevice.DiscoverySession(
      deviceTypes: physicalLensTypes,
      mediaType: .video,
      position: position
    )

    let devices = discovery.devices.sorted { lensSortOrder($0) < lensSortOrder($1) }
    return devices.map { lens -> (zoomFactor: CGFloat, label: String) in
      let zoom = zoomFactor(for: lens, on: activeDevice) ?? fallbackZoomFactor(for: lens)
      return (zoom, lensLabel(for: lens))
    }
  }

  /// Switch to the physical lens whose zoom factor most closely matches
  /// `zoomFactor` (e.g. 0.5 / 1 / 2). Bypasses the 1.0 minimum used by
  /// `setZoomLevel` so the ultra-wide (0.5x) is reachable. Updates the
  /// zoom label, fires `onZoomChanged`, and emits a `lens` event so the
  /// Dart lens picker can sync.
  public func setLens(zoomFactor desired: CGFloat) {
    let lenses = availableLenses()
    guard !lenses.isEmpty else { return }

    // Snap to the lens with the smallest |zoomFactor - desired|.
    let best = lenses.min(by: { abs($0.zoomFactor - desired) < abs($1.zoomFactor - desired) })
    guard let target = best, let device = videoCapture.captureDevice else { return }

    let clamped = min(
      max(target.zoomFactor, device.minAvailableVideoZoomFactor),
      device.maxAvailableVideoZoomFactor
    )

    do {
      try device.lockForConfiguration()
      defer { device.unlockForConfiguration() }
      device.videoZoomFactor = clamped
      lastZoomFactor = clamped
      self.labelZoom.text = String(format: "%.1fx", clamped)
      onZoomChanged?(clamped)
    } catch {
      NSLog("YOLOView: setLens failed: %@", error.localizedDescription)
      return
    }

    // Always emit `lens` for an explicit setLens call so the Dart UI can
    // confirm selection — even when the lens didn't actually change.
    currentLensLabel = target.label
    onLensChanged?(target.label)
  }

  /// Set focus + exposure at a normalized 0..1 view-relative coordinate.
  /// Dart-side gesture handlers call this; no native recognizer is attached
  /// to the view.
  ///
  /// `focusPointOfInterest`/`exposurePointOfInterest` live in the capture
  /// device's coordinate space (aspect-fill cropped, orientation-baked).
  /// View-relative input must therefore be routed through the preview
  /// layer's `captureDevicePointConverted(fromLayerPoint:)` — without that
  /// hop a portrait device focuses well off the tap location.
  public func tapToFocus(x: CGFloat, y: CGFloat) {
    guard let device = videoCapture.captureDevice else { return }
    let viewX = max(0, min(1, x))
    let viewY = max(0, min(1, y))

    // Map 0..1 view coords → preview layer point → capture device point.
    // Falls back to the raw view-relative point if the preview layer is not
    // attached yet (early-frame race), which is the same behavior as iOS
    // before iOS 11 introduced the converter.
    let devicePoint: CGPoint
    if let preview = videoCapture.previewLayer, preview.bounds.width > 0, preview.bounds.height > 0
    {
      let layerPoint = CGPoint(
        x: viewX * preview.bounds.width,
        y: viewY * preview.bounds.height)
      devicePoint = preview.captureDevicePointConverted(fromLayerPoint: layerPoint)
    } else {
      devicePoint = CGPoint(x: viewX, y: viewY)
    }

    do {
      try device.lockForConfiguration()
      defer { device.unlockForConfiguration() }

      if device.isFocusPointOfInterestSupported,
        device.isFocusModeSupported(.autoFocus)
      {
        device.focusPointOfInterest = devicePoint
        device.focusMode = .autoFocus
      }
      if device.isExposurePointOfInterestSupported,
        device.isExposureModeSupported(.autoExpose)
      {
        device.exposurePointOfInterest = devicePoint
        device.exposureMode = .autoExpose
      }
      // Notify Dart with the original view-relative coords so the
      // FocusReticle pulses where the user actually tapped.
      onFocusTapped?(viewX, viewY)
    } catch {
      NSLog("YOLOView: tapToFocus failed: %@", error.localizedDescription)
    }
  }

  /// Returns the user-facing label of the currently selected lens for the
  /// active camera position. Useful for callers that want to seed Dart
  /// state on first frame.
  public func currentLens() -> String {
    if !currentLensLabel.isEmpty { return currentLensLabel }
    guard let device = videoCapture.captureDevice else { return "" }
    return lensLabel(for: device)
  }

  /// Port of `YOLOView.updateSelectedLens` (yolo-ios-app:1157-1185). Picks
  /// the largest-zoom physical lens whose threshold is <= `rawZoomFactor`,
  /// falling back to the smallest available lens. Emits `onLensChanged`
  /// when the label transitions to a new value.
  private func updateSelectedLensLabel(rawZoomFactor: CGFloat, device: AVCaptureDevice) {
    guard device.position == .back else {
      // Front camera: a single device.
      let label = lensLabel(for: device)
      if label != currentLensLabel {
        currentLensLabel = label
        onLensChanged?(label)
      }
      return
    }

    let discovery = AVCaptureDevice.DiscoverySession(
      deviceTypes: physicalLensTypes,
      mediaType: .video,
      position: .back
    )

    let lensZooms = discovery.devices.compactMap {
      (lens: AVCaptureDevice) -> (device: AVCaptureDevice, zoom: CGFloat)? in
      guard physicalLensTypes.contains(lens.deviceType) else { return nil }
      let zoom = zoomFactor(for: lens, on: device) ?? fallbackZoomFactor(for: lens)
      return (lens, zoom)
    }.sorted { $0.zoom < $1.zoom }

    guard !lensZooms.isEmpty else { return }
    let selected =
      lensZooms.last(where: { rawZoomFactor >= $0.zoom - 0.01 })?.device
      ?? lensZooms.first?.device
    guard let selected else { return }

    let label = lensLabel(for: selected)
    if label != currentLensLabel {
      currentLensLabel = label
      onLensChanged?(label)
    }
  }

  /// Lens-label mapping mirroring upstream `lensCaption`.
  private func lensLabel(for device: AVCaptureDevice) -> String {
    if device.position == .front { return "Front camera" }
    switch device.deviceType {
    case .builtInUltraWideCamera: return "Ultra wide camera"
    case .builtInWideAngleCamera: return "Wide camera"
    case .builtInTelephotoCamera: return "Telephoto camera"
    default: return device.localizedName
    }
  }

  private func lensSortOrder(_ device: AVCaptureDevice) -> Int {
    switch device.deviceType {
    case .builtInUltraWideCamera: return 0
    case .builtInWideAngleCamera: return 1
    case .builtInTelephotoCamera: return 2
    default: return 3
    }
  }

  /// Per-lens fallback zoom factor when the active device is not a virtual
  /// multi-lens device (matches the upstream `fallbackLensTitle` numeric
  /// values: 0.5 / 1 / 2).
  private func fallbackZoomFactor(for device: AVCaptureDevice) -> CGFloat {
    switch device.deviceType {
    case .builtInUltraWideCamera: return 0.5
    case .builtInWideAngleCamera: return 1.0
    case .builtInTelephotoCamera: return 2.0
    default: return 1.0
    }
  }

  /// Port of upstream `VideoCapture.swift:62-84` — computes the raw zoom
  /// factor on `virtualDevice` that selects the constituent lens
  /// `lensDevice`.
  private func zoomFactor(
    for lensDevice: AVCaptureDevice, on virtualDevice: AVCaptureDevice?
  ) -> CGFloat? {
    guard let virtualDevice, lensDevice.position == virtualDevice.position else { return nil }
    let constituent = virtualDevice.constituentDevices
      .filter { physicalLensTypes.contains($0.deviceType) }
      .sorted { lensSortOrder($0) < lensSortOrder($1) }
    guard constituent.count > 1 else { return nil }

    let lensIndex =
      constituent.firstIndex { $0.uniqueID == lensDevice.uniqueID }
      ?? constituent.firstIndex { $0.deviceType == lensDevice.deviceType }
    guard let lensIndex else { return nil }

    let switchOverZoomFactors = virtualDevice.virtualDeviceSwitchOverVideoZoomFactors.map {
      CGFloat(truncating: $0)
    }
    let zoomFactors = [virtualDevice.minAvailableVideoZoomFactor] + switchOverZoomFactors
    guard lensIndex < zoomFactors.count else { return nil }

    return min(
      max(zoomFactors[lensIndex], virtualDevice.minAvailableVideoZoomFactor),
      virtualDevice.maxAvailableVideoZoomFactor
    )
  }

  // MARK: - Camera-flip blur transition
  //
  // Ported from `yolo-ios-app/Sources/YOLO/YOLOView.swift:1036-1069`.
  // Adds a snapshot + UIVisualEffectView over the preview while the camera
  // session reconfigures, then fades it out.

  private func showCameraTransition() {
    cameraTransitionView?.removeFromSuperview()

    let transitionView = UIView(frame: bounds)
    transitionView.isUserInteractionEnabled = false
    transitionView.backgroundColor = .black
    transitionView.autoresizingMask = [.flexibleWidth, .flexibleHeight]

    if let snapshot = snapshotView(afterScreenUpdates: false) {
      snapshot.frame = transitionView.bounds
      snapshot.autoresizingMask = [.flexibleWidth, .flexibleHeight]
      transitionView.addSubview(snapshot)
    }

    let blurView = UIVisualEffectView(effect: UIBlurEffect(style: .systemUltraThinMaterialDark))
    blurView.frame = transitionView.bounds
    blurView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
    transitionView.addSubview(blurView)

    // Insert below the top-bar label so any Flutter-side overlays still feel
    // pinned, but above the preview + bounding boxes.
    if labelName.superview === self {
      insertSubview(transitionView, belowSubview: labelName)
    } else {
      addSubview(transitionView)
    }
    cameraTransitionView = transitionView
  }

  private func hideCameraTransition() {
    guard let transitionView = cameraTransitionView else { return }
    cameraTransitionView = nil
    UIView.animate(
      withDuration: 0.18,
      delay: 0.06,
      options: [.beginFromCurrentState, .curveEaseOut]
    ) {
      transitionView.alpha = 0
    } completion: { _ in
      transitionView.removeFromSuperview()
    }
  }

  public func setTorchMode(_ enabled: Bool) {
    guard let device = videoCapture.captureDevice, device.hasTorch else { return }

    do {
      try device.lockForConfiguration()
      defer {
        device.unlockForConfiguration()
      }

      if enabled {
        try device.setTorchModeOn(level: AVCaptureDevice.maxAvailableTorchLevel)
      } else {
        device.torchMode = .off
      }
    } catch {
      NSLog("YOLOView: Failed to set torch mode: %@", error.localizedDescription)
    }
  }

  @objc func playTapped() {
    selection.selectionChanged()
    self.videoCapture.start()
    playButton.isEnabled = false
    pauseButton.isEnabled = true
  }

  @objc func pauseTapped() {
    selection.selectionChanged()
    self.videoCapture.stop()
    playButton.isEnabled = true
    pauseButton.isEnabled = false
  }

  @objc func switchCameraTapped() {

    let authStatus = AVCaptureDevice.authorizationStatus(for: .video)
    if authStatus != .authorized {
      NSLog("YOLOView: Camera permission not authorized. Cannot switch camera.")
      return
    }

    // Visual polish: snapshot+blur the preview while the session
    // reconfigures (port of yolo-ios-app YOLOView.swift:1036-1060).
    showCameraTransition()

    self.videoCapture.captureSession.beginConfiguration()
    guard let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput
    else {
      NSLog("YOLOView: No current camera input to remove")
      self.videoCapture.captureSession.commitConfiguration()
      hideCameraTransition()
      return
    }

    let currentPosition = currentInput.device.position

    self.videoCapture.captureSession.removeInput(currentInput)

    let nextCameraPosition: AVCaptureDevice.Position = currentPosition == .back ? .front : .back

    guard let newCameraDevice = bestCaptureDevice(position: nextCameraPosition) else {
      NSLog(
        "YOLOView: No camera device available for position: %@",
        String(describing: nextCameraPosition))
      self.videoCapture.captureSession.commitConfiguration()
      hideCameraTransition()
      return
    }

    guard let videoInput1 = try? AVCaptureDeviceInput(device: newCameraDevice) else {
      NSLog("YOLOView: Failed to create AVCaptureDeviceInput for camera switch")
      self.videoCapture.captureSession.commitConfiguration()
      hideCameraTransition()
      return
    }

    self.videoCapture.captureSession.addInput(videoInput1)
    self.videoCapture.captureDevice = newCameraDevice
    self.videoCapture.updateVideoOrientation(orientation: currentVideoOrientation())

    self.videoCapture.captureSession.commitConfiguration()

    // Reset lens label cache so the next zoom step (or a getAvailableLenses
    // poll from Dart) reports the new position's lens.
    currentLensLabel = ""
    lastZoomFactor = 1.0

    hideCameraTransition()
  }

  /// Capture a share image. When `withOverlays` is true the next live frame
  /// (or the paused-share frame when the session is stopped) is composited
  /// with the current bounding-box / mask / pose layers via
  /// `renderShareImage`. When false the raw oriented camera frame is
  /// returned so callers can do their own annotation. Matches upstream YOLO
  /// iOS `capturePhoto` and the Android `capturePhoto(withOverlays)` contract.
  public func capturePhoto(withOverlays: Bool, completion: @escaping (UIImage?) -> Void) {
    if let pausedShareImage, !videoCapture.captureSession.isRunning {
      completion(withOverlays ? renderShareImage(pausedShareImage) : pausedShareImage)
      return
    }
    videoCapture.captureNextFrame { [weak self] image in
      guard let self, let image else {
        completion(nil)
        return
      }
      completion(withOverlays ? self.renderShareImage(image) : image)
    }
  }

  public func setInferenceFlag(ok: Bool) {
    videoCapture.inferenceOK = ok
  }

  deinit {
    // Ensure camera is stopped when view is deallocated
    videoCapture.stop()

    // Clear delegate to break retain cycle
    videoCapture.delegate = nil

    // Release predictor to prevent memory leak
    videoCapture.predictor = nil

    // Clear all callbacks to prevent retain cycles
    onDetection = nil
    onStream = nil
    onZoomChanged = nil
    onLensChanged = nil
    onFocusTapped = nil

    // Remove notification observers
    NotificationCenter.default.removeObserver(self)
  }
}

extension YOLOView {
  /// Composites bounding boxes (and mask/pose overlays when present) on top
  /// of a freshly captured frame via `drawHierarchy`. Mirrors upstream YOLO
  /// iOS `renderShareImage`. Mutates the layer hierarchy transiently and
  /// restores it before returning.
  fileprivate func renderShareImage(_ image: UIImage) -> UIImage? {
    var isCameraFront = false
    if let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput,
      currentInput.device.position == .front
    {
      isCameraFront = true
    }
    var orientation: CGImagePropertyOrientation = isCameraFront ? .leftMirrored : .right
    switch UIDevice.current.orientation {
    case .landscapeLeft:
      orientation = isCameraFront ? .downMirrored : .up
    case .landscapeRight:
      orientation = isCameraFront ? .upMirrored : .down
    default:
      break
    }
    var oriented = image
    if let orientedCIImage = CIImage(image: image)?.oriented(orientation),
      let cgImage = CIContext().createCGImage(orientedCIImage, from: orientedCIImage.extent)
    {
      oriented = UIImage(cgImage: cgImage)
    }

    let imageView = UIImageView(image: oriented)
    imageView.contentMode = .scaleAspectFill
    imageView.frame = self.frame
    let imageLayer = imageView.layer
    self.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)

    var tempMaskLayer: CALayer?
    if let maskLayer = self.maskLayer, !maskLayer.isHidden {
      let tempLayer = CALayer()
      let overlayFrame = self.overlayLayer.frame
      let maskFrame = maskLayer.frame
      tempLayer.frame = CGRect(
        x: overlayFrame.origin.x + maskFrame.origin.x,
        y: overlayFrame.origin.y + maskFrame.origin.y,
        width: maskFrame.width,
        height: maskFrame.height
      )
      tempLayer.contents = maskLayer.contents
      tempLayer.contentsGravity = maskLayer.contentsGravity
      tempLayer.contentsRect = maskLayer.contentsRect
      tempLayer.contentsCenter = maskLayer.contentsCenter
      tempLayer.opacity = maskLayer.opacity
      tempLayer.compositingFilter = maskLayer.compositingFilter
      tempLayer.transform = maskLayer.transform
      tempLayer.masksToBounds = maskLayer.masksToBounds
      self.layer.insertSublayer(tempLayer, above: imageLayer)
      tempMaskLayer = tempLayer
    }

    var tempPoseLayer: CALayer?
    if let poseLayer = self.poseLayer {
      let tempLayer = CALayer()
      let overlayFrame = self.overlayLayer.frame
      tempLayer.frame = CGRect(
        x: overlayFrame.origin.x,
        y: overlayFrame.origin.y,
        width: overlayFrame.width,
        height: overlayFrame.height
      )
      tempLayer.opacity = poseLayer.opacity
      if let sublayers = poseLayer.sublayers {
        for sublayer in sublayers {
          let copyLayer = CALayer()
          copyLayer.frame = sublayer.frame
          copyLayer.backgroundColor = sublayer.backgroundColor
          copyLayer.cornerRadius = sublayer.cornerRadius
          copyLayer.opacity = sublayer.opacity
          if let shapeLayer = sublayer as? CAShapeLayer {
            let copyShapeLayer = CAShapeLayer()
            copyShapeLayer.frame = shapeLayer.frame
            copyShapeLayer.path = shapeLayer.path
            copyShapeLayer.strokeColor = shapeLayer.strokeColor
            copyShapeLayer.lineWidth = shapeLayer.lineWidth
            copyShapeLayer.fillColor = shapeLayer.fillColor
            copyShapeLayer.opacity = shapeLayer.opacity
            tempLayer.addSublayer(copyShapeLayer)
          } else {
            tempLayer.addSublayer(copyLayer)
          }
        }
      }
      self.layer.insertSublayer(tempLayer, above: imageLayer)
      tempPoseLayer = tempLayer
    }

    var tempViews = [UIView]()
    let boundingBoxInfos = makeBoundingBoxInfos(from: boundingBoxViews)
    for info in boundingBoxInfos where !info.isHidden {
      let boxView = createBoxView(from: info)
      boxView.frame = info.rect
      self.addSubview(boxView)
      tempViews.append(boxView)
    }

    // Snapshot the YOLOView's own bounds — UIScreen.main.bounds would crop
    // the wrong rect under split view, embedded layouts, or any non-fullscreen
    // host and would misalign overlays in the shared image.
    let bounds = self.bounds
    UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
    drawHierarchy(in: bounds, afterScreenUpdates: true)
    let snapshot = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()

    imageLayer.removeFromSuperlayer()
    tempMaskLayer?.removeFromSuperlayer()
    tempPoseLayer?.removeFromSuperlayer()
    for v in tempViews { v.removeFromSuperview() }

    return snapshot
  }

  // MARK: - Streaming Functionality

  /// Set streaming configuration
  public func setStreamConfig(_ config: YOLOStreamConfig?) {
    self.streamConfig = config
    setupThrottlingFromConfig()
  }

  /// Set streaming callback
  public func setStreamCallback(_ callback: (([String: Any]) -> Void)?) {
    self.onStream = callback
  }

  /// Setup throttling parameters from streaming configuration
  private func setupThrottlingFromConfig() {
    guard let config = streamConfig else { return }

    // Setup maxFPS throttling (for result output)
    if let maxFPS = config.maxFPS, maxFPS > 0 {
      targetFrameInterval = 1.0 / Double(maxFPS)  // Convert to seconds
    } else {
      targetFrameInterval = nil

    }

    // Setup throttleInterval (for result output)
    if let throttleMs = config.throttleIntervalMs, throttleMs > 0 {
      throttleInterval = Double(throttleMs) / 1000.0  // Convert ms to seconds

    } else {
      throttleInterval = nil

    }

    // Setup inference frequency control
    if let inferenceFreq = config.inferenceFrequency, inferenceFreq > 0 {
      inferenceFrameInterval = 1.0 / Double(inferenceFreq)  // Convert to seconds
    } else {
      inferenceFrameInterval = nil
    }

    // Setup frame skipping
    if let skipFrames = config.skipFrames, skipFrames > 0 {
      targetSkipFrames = skipFrames
      frameSkipCount = 0  // Reset counter

    } else {
      targetSkipFrames = 0
      frameSkipCount = 0

    }

    // Initialize timing
    lastInferenceTime = CACurrentMediaTime()
  }

  /// Check if we should run inference on this frame based on inference frequency control
  private func shouldRunInference() -> Bool {
    let now = CACurrentMediaTime()

    // Check frame skipping control first (simpler, more deterministic)
    if targetSkipFrames > 0 {
      frameSkipCount += 1
      if frameSkipCount <= targetSkipFrames {
        // Still skipping frames
        return false
      } else {
        // Reset counter and allow inference
        frameSkipCount = 0
        return true
      }
    }

    // Check inference frequency control (time-based)
    if let interval = inferenceFrameInterval {
      if now - lastInferenceTime < interval {
        return false
      }
    }

    return true
  }

  /// Check if we should send results to Flutter based on output throttling settings
  private func shouldProcessFrame() -> Bool {
    let now = CACurrentMediaTime()

    // Check maxFPS throttling
    if let interval = targetFrameInterval {
      if now - lastInferenceTime < interval {
        return false
      }
    }

    // Check throttleInterval
    if let interval = throttleInterval {
      if now - lastInferenceTime < interval {
        return false
      }
    }

    return true
  }

  /// Update the last inference time (call this when actually processing)
  private func updateLastInferenceTime() {
    lastInferenceTime = CACurrentMediaTime()
  }

  /// Convert YOLOResult to a Dictionary for streaming (ported from Android implementation)
  /// Uses detection index correctly to avoid class index confusion
  private func convertResultToStreamData(_ result: YOLOResult) -> [String: Any] {
    var map: [String: Any] = [:]
    let config = streamConfig ?? YOLOStreamConfig.DEFAULT

    // Convert detection results (if enabled)
    if config.includeDetections {
      var detections: [[String: Any]] = []

      if config.includePoses && !result.keypointsList.isEmpty && result.boxes.isEmpty {
        for (poseIndex, keypoints) in result.keypointsList.enumerated() {
          var detection: [String: Any] = [:]
          detection["classIndex"] = 0
          detection["className"] = "person"
          detection["confidence"] = 1.0
          var minX = Float.greatestFiniteMagnitude
          var minY = Float.greatestFiniteMagnitude
          var maxX = -Float.greatestFiniteMagnitude
          var maxY = -Float.greatestFiniteMagnitude

          for kp in keypoints.xy {
            if kp.x > 0 && kp.y > 0 {
              minX = min(minX, kp.x)
              minY = min(minY, kp.y)
              maxX = max(maxX, kp.x)
              maxY = max(maxY, kp.y)
            }
          }
          let boundingBox: [String: Any] = [
            "left": Double(minX),
            "top": Double(minY),
            "right": Double(maxX),
            "bottom": Double(maxY),
          ]
          detection["boundingBox"] = boundingBox

          // Normalized bounding box
          let normalizedBox: [String: Any] = [
            "left": Double(minX / Float(result.orig_shape.width)),
            "top": Double(minY / Float(result.orig_shape.height)),
            "right": Double(maxX / Float(result.orig_shape.width)),
            "bottom": Double(maxY / Float(result.orig_shape.height)),
          ]
          detection["normalizedBox"] = normalizedBox

          var keypointsFlat: [Double] = []
          for i in 0..<keypoints.xy.count {
            keypointsFlat.append(Double(keypoints.xy[i].x))
            keypointsFlat.append(Double(keypoints.xy[i].y))
            if i < keypoints.conf.count {
              keypointsFlat.append(Double(keypoints.conf[i]))
            } else {
              keypointsFlat.append(0.0)
            }
          }
          detection["keypoints"] = keypointsFlat

          detections.append(detection)
        }
      }

      if !result.obb.isEmpty && result.boxes.isEmpty {
        let imgWidth = result.orig_shape.width
        let imgHeight = result.orig_shape.height

        for obbResult in result.obb {
          var detection: [String: Any] = [:]
          detection["classIndex"] = obbResult.index
          detection["className"] = obbResult.cls
          detection["confidence"] = Double(obbResult.confidence)

          let polygon = obbResult.box.toPolygon(in: result.orig_shape)
          let points = polygon.map { point in
            [
              "x": Double(point.x),
              "y": Double(point.y),
            ]
          }

          var minX = CGFloat.greatestFiniteMagnitude
          var minY = CGFloat.greatestFiniteMagnitude
          var maxX = -CGFloat.greatestFiniteMagnitude
          var maxY = -CGFloat.greatestFiniteMagnitude

          for point in polygon {
            minX = min(minX, point.x)
            minY = min(minY, point.y)
            maxX = max(maxX, point.x)
            maxY = max(maxY, point.y)
          }

          detection["boundingBox"] = [
            "left": Double(minX * imgWidth),
            "top": Double(minY * imgHeight),
            "right": Double(maxX * imgWidth),
            "bottom": Double(maxY * imgHeight),
          ]
          detection["normalizedBox"] = [
            "left": Double(minX),
            "top": Double(minY),
            "right": Double(maxX),
            "bottom": Double(maxY),
          ]

          if config.includeOBB {
            detection["obb"] = [
              "centerX": Double(obbResult.box.cx),
              "centerY": Double(obbResult.box.cy),
              "width": Double(obbResult.box.w),
              "height": Double(obbResult.box.h),
              "angle": Double(obbResult.box.angle),
              "angleDegrees": (Double(obbResult.box.angle) * 180.0 / Double.pi),
              "area": Double(obbResult.box.area),
              "points": points,
              "confidence": Double(obbResult.confidence),
              "className": obbResult.cls,
              "classIndex": obbResult.index,
            ]
          }

          detections.append(detection)
        }
      }

      // Convert detection boxes - CRITICAL: use detectionIndex, not class index
      for (detectionIndex, box) in result.boxes.enumerated() {
        var detection: [String: Any] = [:]
        detection["classIndex"] = box.index
        detection["className"] = box.cls
        detection["confidence"] = Double(box.conf)

        // Bounding box in original coordinates
        let boundingBox: [String: Any] = [
          "left": Double(box.xywh.minX),
          "top": Double(box.xywh.minY),
          "right": Double(box.xywh.maxX),
          "bottom": Double(box.xywh.maxY),
        ]
        detection["boundingBox"] = boundingBox

        // Normalized bounding box (0-1)
        let normalizedBox: [String: Any] = [
          "left": Double(box.xywhn.minX),
          "top": Double(box.xywhn.minY),
          "right": Double(box.xywhn.maxX),
          "bottom": Double(box.xywhn.maxY),
        ]
        detection["normalizedBox"] = normalizedBox

        // Add mask data for segmentation (if available and enabled)
        if config.includeMasks && result.masks?.masks != nil
          && detectionIndex < result.masks!.masks.count
        {
          if let maskData = result.masks?.masks[detectionIndex] {
            // Convert mask data to array format for Flutter compatibility
            let maskDataDouble = maskData.map { row in
              row.map { Double($0) }
            }
            detection["mask"] = maskDataDouble

          }
        }

        // Add pose keypoints (if available and enabled)
        if config.includePoses && detectionIndex < result.keypointsList.count {
          let keypoints = result.keypointsList[detectionIndex]
          // Convert to flat array [x1, y1, conf1, x2, y2, conf2, ...]
          var keypointsFlat: [Double] = []
          for i in 0..<keypoints.xy.count {
            keypointsFlat.append(Double(keypoints.xy[i].x))
            keypointsFlat.append(Double(keypoints.xy[i].y))
            if i < keypoints.conf.count {
              keypointsFlat.append(Double(keypoints.conf[i]))
            } else {
              keypointsFlat.append(0.0)  // Default confidence if missing
            }
          }
          detection["keypoints"] = keypointsFlat
        }

        // Add OBB data (if available and enabled)
        if config.includeOBB && detectionIndex < result.obb.count {
          let obbResult = result.obb[detectionIndex]
          let obbBox = obbResult.box

          // Convert OBB to 4 corner points
          let polygon = obbBox.toPolygon(in: result.orig_shape)
          let points = polygon.map { point in
            [
              "x": Double(point.x),
              "y": Double(point.y),
            ]
          }

          // Create comprehensive OBB data map
          let obbDataMap: [String: Any] = [
            "centerX": Double(obbBox.cx),
            "centerY": Double(obbBox.cy),
            "width": Double(obbBox.w),
            "height": Double(obbBox.h),
            "angle": Double(obbBox.angle),  // radians
            "angleDegrees": (Double(obbBox.angle) * 180.0 / Double.pi),  // degrees for convenience
            "area": Double(obbBox.area),
            "points": points,  // 4 corner points
            "confidence": Double(obbResult.confidence),
            "className": obbResult.cls,
            "classIndex": obbResult.index,
          ]

          detection["obb"] = obbDataMap
        }

        detections.append(detection)
      }
      map["detections"] = detections
    }

    if config.includeMasks, let semanticMask = result.semanticMask {
      map["semanticMask"] = [
        "classMap": semanticMask.classMap,
        "width": semanticMask.width,
        "height": semanticMask.height,
      ]
    }

    // Add classification results (if available and enabled for CLASSIFY task)
    if config.includeClassifications, let probs = result.probs, result.boxes.isEmpty {
      // Get or create detections array (for compatibility with YOLOResult deserialization)
      var detections = map["detections"] as? [[String: Any]] ?? []

      // Build top5 list with labels and confidence
      // Note: iOS native API doesn't provide top5 indices, so we omit class field
      // to maintain consistency with Android when indices are unavailable
      var top5List: [[String: Any]] = []
      let top5Labels = probs.top5Labels
      let top5Confs = probs.top5Confs

      for i in 0..<min(top5Labels.count, top5Confs.count) {
        top5List.append([
          "name": top5Labels[i],
          "confidence": Double(top5Confs[i]),
        ])
      }

      // Create single detection object with top1 and top5 info
      var detection: [String: Any] = [:]

      // Note: iOS native API doesn't provide real class index, so we omit it
      // to avoid sending potentially incorrect values (e.g., -1 or wrong index)
      detection["name"] = probs.top1Label
      detection["confidence"] = Double(probs.top1Conf)
      detection["top5"] = top5List

      // Classification doesn't have bounding boxes, use full image bounds
      let boundingBox: [String: Any] = [
        "left": 0.0,
        "top": 0.0,
        "right": Double(result.orig_shape.width),
        "bottom": Double(result.orig_shape.height),
      ]
      detection["boundingBox"] = boundingBox

      // Normalized bounding box (full image)
      let normalizedBox: [String: Any] = [
        "left": 0.0,
        "top": 0.0,
        "right": 1.0,
        "bottom": 1.0,
      ]
      detection["normalizedBox"] = normalizedBox

      detections.append(detection)
      map["detections"] = detections
    }

    // Add performance metrics (if enabled)
    if config.includeProcessingTimeMs {
      map["processingTimeMs"] = result.speed * 1000
    }

    if config.includeFps {
      map["fps"] = result.fps ?? 0.0
    }

    if config.includeOriginalImage {
      if let originalImage = result.originalImage {
        if let imageData = originalImage.jpegData(compressionQuality: 0.9) {
          map["originalImage"] = imageData
        }
      }
    }

    return map
  }

}
