// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Flutter
import Foundation
import UIKit
import UltralyticsYOLO

/// Manages multiple YOLO instances with unique IDs
@MainActor
class YOLOInstanceManager {
  static let shared = YOLOInstanceManager()

  private var instances: [String: YOLO] = [:]
  private var loadingStates: [String: Bool] = [:]
  private var loadCompletionHandlers: [String: [(Result<YOLO, Error>) -> Void]] = [:]

  private init() {
    createInstance(instanceId: "default")
  }

  /// Creates a new YOLO instance with the given ID
  func createInstance(instanceId: String) {
    loadCompletionHandlers[instanceId] = []
    loadingStates[instanceId] = false
  }

  /// Gets a YOLO instance by ID
  func getInstance(instanceId: String) -> YOLO? {
    return instances[instanceId]
  }

  /// Loads a model for a specific instance
  func loadModel(
    instanceId: String,
    modelName: String,
    task: YOLOTask,
    useGpu: Bool = true,
    numItemsThreshold: Int = 30,
    completion: @escaping (Result<Void, Error>) -> Void
  ) {
    // Check if model is already loaded
    if instances[instanceId] != nil {
      completion(.success(()))
      return
    }

    // Check if loading is in progress
    if loadingStates[instanceId] == true {
      loadCompletionHandlers[instanceId]?.append({ result in
        switch result {
        case .success:
          completion(.success(()))
        case .failure(let error):
          completion(.failure(error))
        }
      })
      return
    }

    // Start loading
    loadingStates[instanceId] = true

    let resolvedModelPath = resolveModelPath(modelName)

    // YOLO retains itself through its load-completion closure until the model finishes loading, so silencing the
    // "Result of 'YOLO' initializer is unused" warning by binding to `_` is sufficient.
    _ = YOLO(resolvedModelPath, task: task, useGpu: useGpu, numItemsThreshold: numItemsThreshold) {
      [weak self] result in
      guard let self = self else { return }

      self.loadingStates[instanceId] = false

      switch result {
      case .success(let loadedYolo):
        self.instances[instanceId] = loadedYolo
        completion(.success(()))

        // Call all pending handlers
        if let handlers = self.loadCompletionHandlers[instanceId] {
          for handler in handlers {
            handler(.success(loadedYolo))
          }
        }

      case .failure(let error):
        completion(.failure(error))

        // Call all pending handlers with error
        if let handlers = self.loadCompletionHandlers[instanceId] {
          for handler in handlers {
            handler(.failure(error))
          }
        }
      }

      self.loadCompletionHandlers[instanceId]?.removeAll()
    }
  }

  /// Runs inference on a specific instance
  func predict(
    instanceId: String,
    imageData: Data,
    confidenceThreshold: Double? = nil,
    iouThreshold: Double? = nil
  ) -> [String: Any]? {
    guard let yolo = instances[instanceId] else {
      return nil
    }

    guard let image = UIImage(data: imageData) else {
      return nil
    }

    let result: YOLOResult

    // Store original thresholds
    let originalConfThreshold = yolo.getConfidenceThreshold() ?? 0.25
    let originalIouThreshold = yolo.getIouThreshold() ?? 0.7

    // Apply custom thresholds if provided
    if let confThreshold = confidenceThreshold {
      yolo.setConfidenceThreshold(confThreshold)
    }
    if let iouThres = iouThreshold {
      yolo.setIouThreshold(iouThres)
    }

    result = yolo.callAsFunction(image)

    // Restore original thresholds
    yolo.setConfidenceThreshold(originalConfThreshold)
    yolo.setIouThreshold(originalIouThreshold)

    return convertToFlutterFormat(result: result)
  }

  /// Removes an instance
  func removeInstance(instanceId: String) {
    instances.removeValue(forKey: instanceId)
    loadingStates.removeValue(forKey: instanceId)
    loadCompletionHandlers.removeValue(forKey: instanceId)
  }

  /// Gets all active instance IDs
  func getActiveInstanceIds() -> [String] {
    return Array(instances.keys)
  }

  /// Checks if an instance exists
  func hasInstance(instanceId: String) -> Bool {
    return instances[instanceId] != nil
  }

  // MARK: - Private Helpers

  private func resolveModelPath(_ modelPath: String) -> String {
    // Already an absolute path
    if modelPath.hasPrefix("/") {
      return modelPath
    }

    if modelPath.contains("/") {
      let components = modelPath.components(separatedBy: "/")
      let fileName = components.last ?? ""
      let fileNameWithoutExt = fileName.components(separatedBy: ".").first ?? fileName
      let directory = components.dropLast().joined(separator: "/")

      let searchPaths = [
        "flutter_assets/\(modelPath)",
        "flutter_assets/\(directory)",
        "flutter_assets",
        "",
      ]

      for searchPath in searchPaths {
        // Search with full name
        if !searchPath.isEmpty,
          let assetPath = Bundle.main.path(
            forResource: fileName, ofType: nil, inDirectory: searchPath)
        {
          return assetPath
        }

        if fileName.contains(".") {
          let fileComponents = fileName.components(separatedBy: ".")
          let name = fileComponents.dropLast().joined(separator: ".")
          let ext = fileComponents.last ?? ""

          // Search with name and extension
          if !searchPath.isEmpty,
            let assetPath = Bundle.main.path(
              forResource: name, ofType: ext, inDirectory: searchPath)
          {
            return assetPath
          } else if searchPath.isEmpty,
            let assetPath = Bundle.main.path(forResource: name, ofType: ext)
          {
            return assetPath
          }
        }

        // Search without extension
        if !searchPath.isEmpty,
          let assetPath = Bundle.main.path(
            forResource: fileNameWithoutExt, ofType: nil, inDirectory: searchPath)
        {
          return assetPath
        }
      }
    } else {
      // No directory path, search in bundle
      let fileName = modelPath
      let fileNameWithoutExt = fileName.components(separatedBy: ".").first ?? fileName

      // Search in flutter_assets first
      if let assetPath = Bundle.main.path(
        forResource: fileName, ofType: nil, inDirectory: "flutter_assets")
      {
        return assetPath
      }

      if fileName.contains(".") {
        let fileComponents = fileName.components(separatedBy: ".")
        let name = fileComponents.dropLast().joined(separator: ".")
        let ext = fileComponents.last ?? ""

        // Search with name and extension in flutter_assets
        if let assetPath = Bundle.main.path(
          forResource: name, ofType: ext, inDirectory: "flutter_assets")
        {
          return assetPath
        }

        // Search in main bundle
        if let assetPath = Bundle.main.path(forResource: name, ofType: ext) {
          return assetPath
        }
      }

      // Search without extension in flutter_assets
      if let assetPath = Bundle.main.path(
        forResource: fileNameWithoutExt, ofType: nil, inDirectory: "flutter_assets")
      {
        return assetPath
      }

      // Search in main bundle
      if let assetPath = Bundle.main.path(forResource: fileName, ofType: nil) {
        return assetPath
      }

      if let assetPath = Bundle.main.path(forResource: fileNameWithoutExt, ofType: nil) {
        return assetPath
      }
    }

    // Return original path if not found
    return modelPath
  }

  private func convertToFlutterFormat(result: YOLOResult) -> [String: Any] {
    var flutterBoxes: [[String: Any]] = []

    let imageWidth = result.orig_shape.width
    let imageHeight = result.orig_shape.height

    // Convert boxes to Flutter format
    for box in result.boxes {
      let boxDict: [String: Any] = [
        "class": box.cls,
        "className": box.cls,  // Add className for compatibility with YOLOResult
        "confidence": box.conf,
        "x1": box.xywh.minX,
        "y1": box.xywh.minY,
        "x2": box.xywh.maxX,
        "y2": box.xywh.maxY,
        "x1_norm": box.xywh.minX / imageWidth,
        "y1_norm": box.xywh.minY / imageHeight,
        "x2_norm": box.xywh.maxX / imageWidth,
        "y2_norm": box.xywh.maxY / imageHeight,
      ]

      flutterBoxes.append(boxDict)
    }

    var resultDict: [String: Any] = [
      "boxes": flutterBoxes,
      "imageSize": [
        "width": Int(imageWidth),
        "height": Int(imageHeight),
      ],
    ]

    // Pose estimation - keypoints
    if !result.keypointsList.isEmpty {
      var keypointsArray: [[String: Any]] = []

      for keypoints in result.keypointsList {
        var coordinates: [[String: Any]] = []

        for (index, (x, y)) in keypoints.xy.enumerated() {
          if index < keypoints.conf.count {
            coordinates.append([
              "x": x,
              "y": y,
              "confidence": keypoints.conf[index],
            ])
          }
        }

        keypointsArray.append([
          "coordinates": coordinates
        ])
      }

      resultDict["keypoints"] = keypointsArray
    }

    // Classification - probs
    // Use Results.summary() format: {name, class, confidence}
    // Reference: https://docs.ultralytics.com/reference/engine/results
    if let probs = result.probs {
      resultDict["classification"] = [
        "name": probs.top1Label,
        "class": result.names.firstIndex(of: probs.top1Label) ?? 0,
        "confidence": probs.top1Conf,
        "top5": probs.top5Labels.enumerated().map { (index, label) in
          [
            "name": label,
            "confidence": index < probs.top5Confs.count ? probs.top5Confs[index] : 0.0,
          ]
        },
      ]
    }

    // Segmentation - masks
    if let masks = result.masks {
      // Send raw mask data for each detected instance
      var rawMasks: [[[Double]]] = []

      for instanceMask in masks.masks {
        var mask2D: [[Double]] = []
        for row in instanceMask {
          mask2D.append(row.map { Double($0) })
        }
        rawMasks.append(mask2D)
      }
      resultDict["masks"] = rawMasks

      // Also send PNG for backward compatibility (optional)
      if let combinedMask = masks.combinedMask {
        let ciImage = CIImage(cgImage: combinedMask)
        let context = CIContext()
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
          let uiImage = UIImage(cgImage: cgImage)
          if let maskData = uiImage.pngData() {
            resultDict["maskPng"] = FlutterStandardTypedData(bytes: maskData)
          }
        }
      }
    }

    if let semanticMask = result.semanticMask {
      resultDict["semanticMask"] = [
        "classMap": semanticMask.classMap,
        "width": semanticMask.width,
        "height": semanticMask.height,
      ]
    }

    // OBB - oriented bounding boxes
    if !result.obb.isEmpty {
      var obbArray: [[String: Any]] = []

      for obbResult in result.obb {
        let box = obbResult.box
        let points = box.toPolygon(in: result.orig_shape).map { point in
          ["x": Float(point.x), "y": Float(point.y)]
        }

        obbArray.append([
          "points": points,
          "angle": Double(box.angle),
          "classIndex": obbResult.index,
          "class": obbResult.cls,
          "confidence": obbResult.confidence,
        ])
      }

      resultDict["obb"] = obbArray
    }

    // Include annotated image if available
    if let annotatedImage = result.annotatedImage {
      if let imageData = annotatedImage.pngData() {
        resultDict["annotatedImage"] = FlutterStandardTypedData(bytes: imageData)
      }
    }

    // Include speed metric
    resultDict["speed"] = result.speed

    return resultDict
  }
}
