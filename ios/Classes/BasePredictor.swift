// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Ultralytics YOLO Package, providing the base infrastructure for model prediction.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The BasePredictor class is the foundation for all task-specific predictors in the YOLO framework.
//  It manages the loading and initialization of Core ML models, handling common operations such as
//  model loading, class label extraction, and inference timing. The class provides an asynchronous
//  model loading mechanism that runs on background threads and includes support for configuring
//  model parameters like confidence thresholds and IoU thresholds. Specific task implementations
//  (detection, segmentation, classification, etc.) inherit from this base class and override
//  the prediction-specific methods.

import CoreImage
import Foundation
import UIKit
import Vision

/// Base class for all YOLO model predictors, handling common model loading and inference logic.
///
/// The BasePredictor serves as the foundation for all task-specific YOLO model predictors.
/// It manages Core ML model loading, initialization, and common inference operations.
/// Specialized predictors (for detection, segmentation, etc.) inherit from this class
/// and override the prediction-specific methods to handle task-specific processing.
///
/// - Note: This class is marked as `@unchecked Sendable` to support concurrent operations.
/// - Important: Task-specific implementations must override the `processObservations` and
///   `predictOnImage` methods to provide proper functionality.
public class BasePredictor: Predictor, @unchecked Sendable {
  /// Flag indicating if the model has been successfully loaded and is ready for inference.
  private(set) var isModelLoaded: Bool = false

  /// The Vision Core ML model used for inference operations.
  var detector: VNCoreMLModel!

  /// The Vision request that processes images using the Core ML model.
  var visionRequest: VNCoreMLRequest?

  /// Vision preprocessing mode for this predictor.
  var imageCropAndScaleOption: VNImageCropAndScaleOption { .scaleFit }

  /// The class labels used by the model for categorizing detections.
  public var labels = [String]()

  /// Whether the model needs external NMS. NMS-free exports (YOLO26) set metadata `nms = false`; for those the Vision
  /// ThresholdProvider IoU is forced to 1.0 so the built-in NMS-free decoding isn't suppressed. Mirrors yolo-ios-app.
  public private(set) var requiresNMS: Bool = true

  /// The current pixel buffer being processed (used for camera frame processing).
  var currentBuffer: CVPixelBuffer?

  /// The current listener to receive prediction results.
  weak var currentOnResultsListener: ResultsListener?

  /// The current listener to receive inference timing information.
  weak var currentOnInferenceTimeListener: InferenceTimeListener?

  /// The size of the input image or camera frame.
  var inputSize: CGSize!

  /// The required input dimensions for the model (width and height in pixels).
  var modelInputSize: (width: Int, height: Int) = (0, 0)

  /// Timestamp for the start of inference (used for performance measurement).
  var t0 = 0.0  // inference start

  /// Duration of a single inference operation.
  var t1 = 0.0  // inference dt

  /// Smoothed inference duration (averaged over recent operations).
  var t2 = 0.0  // inference dt smoothed

  /// Timestamp for FPS calculation start (used for performance measurement).
  var t3 = CACurrentMediaTime()  // FPS start

  /// Smoothed frames per second measurement (averaged over recent frames).
  var t4 = 0.0  // FPS dt smoothed

  /// Flag indicating whether the predictor is currently processing an update.
  public var isUpdating: Bool = false

  /// Stream configuration for controlling what data is included in results.
  var streamConfig: YOLOStreamConfig?

  /// Original image data captured for streaming (if enabled).
  var originalImageData: Data?

  /// Required initializer for creating predictor instances.
  ///
  /// This empty initializer is required for the factory pattern used in the `create` method.
  /// Subclasses may override this to perform additional initialization.
  required init() {
    // Intentionally left empty
  }

  func labelName(for index: Int) -> String {
    guard index >= 0 else { return "class_\(index)" }
    guard index < labels.count else { return "class_\(index)" }

    let label = labels[index].trimmingCharacters(in: .whitespacesAndNewlines)
    return label.isEmpty ? "class_\(index)" : label
  }

  private static func parseLabels(from userDefined: [String: String]) -> [String] {
    if let labelsData = userDefined["classes"] {
      return
        labelsData
        .components(separatedBy: ",")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    }

    if let labelsData = userDefined["names"] {
      let cleanedInput =
        labelsData
        .replacingOccurrences(of: "{", with: "")
        .replacingOccurrences(of: "}", with: "")

      let parsedPairs = cleanedInput.components(separatedBy: ",").compactMap {
        pair -> (Int?, String)? in
        let components = pair.split(
          separator: ":",
          maxSplits: 1,
          omittingEmptySubsequences: false
        )
        guard components.count >= 2 else { return nil }

        let key = Int(String(components[0]).trimmingCharacters(in: .whitespacesAndNewlines))
        let value = String(components[1])
          .trimmingCharacters(in: .whitespacesAndNewlines)
          .replacingOccurrences(of: "'", with: "")
        return (key, value)
      }

      let keyedLabels = parsedPairs.compactMap { key, value -> (Int, String)? in
        guard let key else { return nil }
        return (key, value)
      }
      if !keyedLabels.isEmpty {
        let maxKey = keyedLabels.map(\.0).max() ?? -1
        var labels = Array(repeating: "", count: maxKey + 1)
        for (key, value) in keyedLabels {
          labels[key] = value
        }
        return labels
      }

      return parsedPairs.map { $0.1 }
    }

    return []
  }

  /// Cancels any pending Vision requests and releases references on deinit.
  deinit {
    visionRequest?.cancel()
    visionRequest = nil
    detector = nil
    currentBuffer = nil
    currentOnResultsListener = nil
    currentOnInferenceTimeListener = nil
  }

  /// Factory method to asynchronously create and initialize a predictor with the specified model.
  ///
  /// This method loads the Core ML model in a background thread and sets up the prediction
  /// infrastructure. The completion handler is called on the main thread with either a
  /// successfully initialized predictor or an error.
  ///
  /// - Parameters:
  ///   - unwrappedModelURL: The URL of the Core ML model file to load.
  ///   - isRealTime: Flag indicating if the predictor will be used for real-time processing (camera feed).
  ///   - useGpu: Flag indicating whether to use GPU acceleration
  ///   - completion: Callback that receives the initialized predictor or an error.
  /// - Note: Model loading happens on a background thread to avoid blocking the main thread.
  public static func create(
    unwrappedModelURL: URL,
    isRealTime: Bool = false,
    useGpu: Bool = true,
    numItemsThreshold: Int = 30,
    completion: @escaping (Result<BasePredictor, Error>) -> Void
  ) {
    let predictor = Self.init()
    predictor.numItemsThreshold = numItemsThreshold

    // Carry the non-Sendable completion across queue boundaries via SendableBox so Swift 6 strict concurrency
    // doesn't flag the main-queue dispatch below; all invocations land on the main queue.
    let completionBox = SendableBox(completion)
    // Kick off the expensive loading on a background thread
    DispatchQueue.global(qos: .userInitiated).async {
      do {
        // (1) Load the MLModel
        let ext = unwrappedModelURL.pathExtension.lowercased()
        let isCompiled = (ext == "mlmodelc")
        let config = MLModelConfiguration()

        if useGpu {
          // Enable GPU acceleration
          config.computeUnits = .all
        } else {
          // Avoid GPU while keeping Neural Engine acceleration available.
          if #available(iOS 16.0, *) {
            config.computeUnits = .cpuAndNeuralEngine
          } else {
            config.computeUnits = .cpuOnly
          }
        }

        if #available(iOS 17.0, *) {
          config.setValue(1, forKey: "experimentalMLE5EngineUsage")
        }

        let mlModel: MLModel
        if isCompiled {
          mlModel = try MLModel(contentsOf: unwrappedModelURL, configuration: config)
        } else {
          let compiledUrl = try MLModel.compileModel(at: unwrappedModelURL)
          mlModel = try MLModel(contentsOf: compiledUrl, configuration: config)
        }

        let userDefined =
          mlModel.modelDescription
          .metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String]

        // Continue even when top-level metadata is missing. Some Core ML pipeline exports
        // only keep labels on nested models, and hard-failing here leaves the predictor nil.
        predictor.labels = userDefined.map(Self.parseLabels(from:)) ?? []

        // Detect NMS-free models (YOLO26): metadata `nms` == "false".
        if let nmsValue = userDefined?["nms"] {
          predictor.requiresNMS = (nmsValue.lowercased() != "false")
        }

        // (3) Store model input size
        predictor.modelInputSize = predictor.getModelInputSize(for: mlModel)

        // (4) Create VNCoreMLModel, VNCoreMLRequest, etc.
        predictor.detector = try VNCoreMLModel(for: mlModel)
        // Seed the model's threshold inputs at load (NMS-free models force IoU = 1.0 so their decoding isn't
        // suppressed). Previously a bare ThresholdProvider() left default thresholds until the user moved a slider.
        let seedIou = predictor.requiresNMS ? predictor.iouThreshold : 1.0
        predictor.detector.featureProvider = ThresholdProvider(
          iouThreshold: seedIou, confidenceThreshold: predictor.confidenceThreshold)
        predictor.visionRequest = {
          let request = VNCoreMLRequest(
            model: predictor.detector,
            completionHandler: {
              [weak predictor] request, error in
              guard let predictor = predictor else {
                // The predictor was deallocated — do nothing
                return
              }
              if isRealTime {
                predictor.processObservations(for: request, error: error)
              }
            })
          request.imageCropAndScaleOption = predictor.imageCropAndScaleOption
          return request
        }()

        // Once done, mark it loaded
        predictor.isModelLoaded = true

        // Finally, call the completion on the main thread
        let predictorBox = SendableBox(predictor)
        DispatchQueue.main.async {
          completionBox.value(.success(predictorBox.value))
        }
      } catch {
        // If anything goes wrong, call completion with the error
        let errorBox = SendableBox(error)
        DispatchQueue.main.async {
          completionBox.value(.failure(errorBox.value))
        }
      }
    }
  }

  /// Processes a camera frame buffer and delivers results via callbacks.
  ///
  /// This method takes a camera sample buffer, performs inference using the Vision framework,
  /// and notifies listeners with the results and performance metrics. It's designed to be
  /// called repeatedly with frames from a camera feed.
  ///
  /// - Parameters:
  ///   - sampleBuffer: The camera frame buffer to process.
  ///   - onResultsListener: Optional listener to receive prediction results.
  ///   - onInferenceTime: Optional listener to receive performance metrics.
  func predict(
    sampleBuffer: CMSampleBuffer, onResultsListener: ResultsListener?,
    onInferenceTime: InferenceTimeListener?
  ) {
    if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      currentBuffer = pixelBuffer
      inputSize = CGSize(
        width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
      currentOnResultsListener = onResultsListener
      currentOnInferenceTimeListener = onInferenceTime
      //            currentOnFpsRateListener = onFpsRate

      /// - Tag: MappingOrientation
      // The frame is always oriented based on the camera sensor,
      // so in most cases Vision needs to rotate it for the model to work as expected.
      let imageOrientation: CGImagePropertyOrientation = .up

      // Capture original image data for streaming if needed
      let originalImageData: Data? =
        streamConfig?.includeOriginalImage == true
        ? convertPixelBufferToJPEGData(pixelBuffer)
        : nil

      // Invoke a VNRequestHandler with that image
      let handler = VNImageRequestHandler(
        cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
      self.originalImageData = originalImageData
      t0 = CACurrentMediaTime()  // inference start
      do {
        if let request = visionRequest {
          try handler.perform([request])
        }
      } catch {
        NSLog("YOLO inference error: %@", String(describing: error))
      }
      t1 = CACurrentMediaTime() - t0  // inference dt

      currentBuffer = nil
    }
  }

  /// Updates inference and FPS timing with the current frame measurements.
  @discardableResult
  func updateTiming() -> (speed: Double, fps: Double) {
    let now = CACurrentMediaTime()
    let inferenceTime = t0 > 0 ? now - t0 : t1
    t1 = inferenceTime

    if inferenceTime < 10.0 {
      t2 = t2 == 0 ? inferenceTime : inferenceTime * 0.05 + t2 * 0.95
    }

    let frameInterval = now - t3
    if frameInterval > 0, frameInterval < 10.0 {
      t4 = t4 == 0 ? frameInterval : frameInterval * 0.05 + t4 * 0.95
    }
    t3 = now

    let fps = t4 > 0 ? 1 / t4 : 0
    currentOnInferenceTimeListener?.on(inferenceTime: t2 * 1000, fpsRate: fps)
    return (speed: t2, fps: fps)
  }

  /// The confidence threshold for filtering detection results (default: 0.25).
  ///
  /// Only detections with confidence scores above this threshold will be included in results.
  var confidenceThreshold = 0.25

  /// Sets the confidence threshold for filtering results.
  ///
  /// - Parameter confidence: The new confidence threshold value (0.0 to 1.0).
  func setConfidenceThreshold(confidence: Double) {
    confidenceThreshold = confidence
  }

  /// The IoU (Intersection over Union) threshold for non-maximum suppression (default: 0.7).
  ///
  /// Used to filter overlapping detections during non-maximum suppression.
  var iouThreshold = 0.7

  /// Sets the IoU threshold for non-maximum suppression.
  ///
  /// - Parameter iou: The new IoU threshold value (0.0 to 1.0).
  func setIouThreshold(iou: Double) {
    iouThreshold = iou
  }

  /// The maximum number of detections to return in results (default: 30).
  ///
  /// Limits the number of detection items in the final results to prevent overwhelming processing.
  var numItemsThreshold = 30

  /// Sets the maximum number of detection items to include in results.
  ///
  /// - Parameter numItems: The maximum number of items to include.
  func setNumItemsThreshold(numItems: Int) {
    numItemsThreshold = numItems
  }

  /// Processes Vision framework observations from model inference.
  ///
  /// This method is called when Vision completes a request with the model's outputs.
  /// Subclasses must override this method to implement task-specific processing of the
  /// model's output features (e.g., parsing detection boxes, segmentation masks, etc.).
  ///
  /// - Parameters:
  ///   - request: The completed Vision request containing model outputs.
  ///   - error: Any error that occurred during the Vision request.
  func processObservations(for request: VNRequest, error: Error?) {
    // Base implementation is empty - must be overridden by subclasses
  }

  /// Processes a static image and returns results synchronously.
  ///
  /// This method performs model inference on a static image and returns the results.
  /// Subclasses must override this method to implement task-specific processing.
  ///
  /// - Parameter image: The CIImage to process.
  /// - Returns: A YOLOResult containing the prediction outputs.
  func predictOnImage(image: CIImage) -> YOLOResult {
    // Base implementation returns an empty result - must be overridden by subclasses
    return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
  }

  /// Extracts the required input dimensions from the model description.
  ///
  /// This utility method determines the expected input size for the Core ML model
  /// by examining its input description, which is essential for properly sizing
  /// and formatting images before inference.
  ///
  /// - Parameter model: The Core ML model to analyze.
  /// - Returns: A tuple containing the width and height in pixels required by the model.
  func getModelInputSize(for model: MLModel) -> (width: Int, height: Int) {
    guard let inputDescription = model.modelDescription.inputDescriptionsByName.first?.value else {
      return (0, 0)
    }

    if let multiArrayConstraint = inputDescription.multiArrayConstraint {
      let shape = multiArrayConstraint.shape
      if shape.count >= 2 {
        let height = shape[shape.count - 2].intValue
        let width = shape[shape.count - 1].intValue
        return (width: width, height: height)
      }
    }

    if let imageConstraint = inputDescription.imageConstraint {
      return (
        width: Int(imageConstraint.pixelsWide), height: Int(imageConstraint.pixelsHigh)
      )
    }

    return (0, 0)
  }

  private func letterboxTransform() -> (
    gain: CGFloat, padX: CGFloat, padY: CGFloat, padRight: CGFloat, padBottom: CGFloat
  )? {
    let modelWidth = CGFloat(modelInputSize.width)
    let modelHeight = CGFloat(modelInputSize.height)
    let inputWidth = inputSize.width
    let inputHeight = inputSize.height
    guard modelWidth > 0, modelHeight > 0, inputWidth > 0, inputHeight > 0 else { return nil }

    let gain = min(modelWidth / inputWidth, modelHeight / inputHeight)
    guard gain > 0 else { return nil }
    let resizedWidth = (inputWidth * gain).rounded()
    let resizedHeight = (inputHeight * gain).rounded()
    // Match Ultralytics LetterBox leading-pad rounding: round(d - 0.1).
    let padWidth = modelWidth - resizedWidth
    let padHeight = modelHeight - resizedHeight
    let padX = (padWidth / 2 - 0.1).rounded()
    let padY = (padHeight / 2 - 0.1).rounded()
    let padRight = (padWidth / 2 + 0.1).rounded()
    let padBottom = (padHeight / 2 + 0.1).rounded()
    return (gain, padX, padY, padRight, padBottom)
  }

  func modelMaskCropRect(maskWidth: Int, maskHeight: Int) -> CGRect? {
    guard let transform = letterboxTransform() else { return nil }
    let modelWidth = CGFloat(modelInputSize.width)
    let modelHeight = CGFloat(modelInputSize.height)
    let width = CGFloat(maskWidth)
    let height = CGFloat(maskHeight)
    let left = (transform.padX / modelWidth * width).rounded()
    let top = (transform.padY / modelHeight * height).rounded()
    let right = width - (transform.padRight / modelWidth * width).rounded()
    let bottom = height - (transform.padBottom / modelHeight * height).rounded()
    let rect = CGRect(x: left, y: top, width: right - left, height: bottom - top)
      .intersection(CGRect(x: 0, y: 0, width: width, height: height))
    guard rect.width > 0, rect.height > 0 else { return nil }
    return rect == CGRect(x: 0, y: 0, width: width, height: height) ? nil : rect
  }

  func inputRect(fromModelRect rect: CGRect) -> CGRect {
    guard let transform = letterboxTransform() else { return rect }
    let x1 = (rect.minX - transform.padX) / transform.gain
    let y1 = (rect.minY - transform.padY) / transform.gain
    let x2 = (rect.maxX - transform.padX) / transform.gain
    let y2 = (rect.maxY - transform.padY) / transform.gain

    let minX = min(max(min(x1, x2), 0), inputSize.width)
    let minY = min(max(min(y1, y2), 0), inputSize.height)
    let maxX = min(max(max(x1, x2), 0), inputSize.width)
    let maxY = min(max(max(y1, y2), 0), inputSize.height)
    return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
  }

  func normalizedRect(fromInputRect rect: CGRect) -> CGRect {
    guard inputSize.width > 0, inputSize.height > 0 else { return rect }
    return CGRect(
      x: rect.minX / inputSize.width,
      y: rect.minY / inputSize.height,
      width: rect.width / inputSize.width,
      height: rect.height / inputSize.height)
  }

  func inputPoint(fromModelPoint point: CGPoint) -> CGPoint {
    guard let transform = letterboxTransform() else { return point }
    let x = (point.x - transform.padX) / transform.gain
    let y = (point.y - transform.padY) / transform.gain
    return CGPoint(
      x: min(max(x, 0), inputSize.width),
      y: min(max(y, 0), inputSize.height))
  }

  func normalizedPoint(fromInputPoint point: CGPoint) -> CGPoint {
    guard inputSize.width > 0, inputSize.height > 0 else { return point }
    return CGPoint(x: point.x / inputSize.width, y: point.y / inputSize.height)
  }

  func inputOBB(fromModelOBB box: OBB) -> OBB {
    guard let transform = letterboxTransform(), inputSize.width > 0, inputSize.height > 0 else {
      return box
    }
    let modelWidth = CGFloat(modelInputSize.width)
    let modelHeight = CGFloat(modelInputSize.height)
    let centerX = (CGFloat(box.cx) * modelWidth - transform.padX) / transform.gain
    let centerY = (CGFloat(box.cy) * modelHeight - transform.padY) / transform.gain
    let width = CGFloat(box.w) * modelWidth / transform.gain
    let height = CGFloat(box.h) * modelHeight / transform.gain
    return OBB(
      cx: Float(centerX / inputSize.width),
      cy: Float(centerY / inputSize.height),
      w: Float(width / inputSize.width),
      h: Float(height / inputSize.height),
      angle: box.angle)
  }

  /// Convert CVPixelBuffer to JPEG data for streaming
  private func convertPixelBufferToJPEGData(_ pixelBuffer: CVPixelBuffer) -> Data? {
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let context = CIContext()
    guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
    let uiImage = UIImage(cgImage: cgImage)
    return uiImage.jpegData(compressionQuality: 0.9)
  }
}
