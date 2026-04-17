// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Ultralytics YOLO Package, providing the main entry point for using YOLO models.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLO class serves as the primary interface for loading and using YOLO machine learning models.
//  It supports a variety of input formats including UIImage, CIImage, CGImage, and resource files.
//  The class handles model loading, format conversion, and inference execution, offering a simple yet
//  powerful API through Swift's callable object pattern. Users can load models from local bundles or
//  file paths and perform inference with a single function call syntax, making integration into iOS
//  applications straightforward.

import Foundation
import SwiftUI
import UIKit

/// The primary interface for working with YOLO models, supporting multiple input types and inference methods.
public class YOLO {
  var predictor: Predictor!

  /// Confidence threshold for filtering predictions (0.0-1.0)
  public var confidenceThreshold: Double = 0.25 {
    didSet {
      // Apply to predictor if it has been loaded
      if let basePredictor = predictor as? BasePredictor {
        basePredictor.setConfidenceThreshold(confidence: confidenceThreshold)
      }
    }
  }

  /// IoU threshold for non-maximum suppression (0.0-1.0)
  public var iouThreshold: Double = 0.7 {
    didSet {
      // Apply to predictor if it has been loaded
      if let basePredictor = predictor as? BasePredictor {
        basePredictor.setIouThreshold(iou: iouThreshold)
      }
    }
  }

  public init(
    _ modelPathOrName: String, task: YOLOTask, useGpu: Bool = true,
    numItemsThreshold: Int = 30,
    completion: ((Result<YOLO, Error>) -> Void)? = nil
  ) {
    var modelURL: URL?

    let lowercasedPath = modelPathOrName.lowercased()
    let fileManager = FileManager.default

    // Check absolute paths, including mlpackage directories
    if lowercasedPath.hasSuffix(".mlmodel") || lowercasedPath.hasSuffix(".mlpackage") {
      let possibleURL = URL(fileURLWithPath: modelPathOrName)
      var isDirectory: ObjCBool = false
      if fileManager.fileExists(atPath: possibleURL.path, isDirectory: &isDirectory) {
        // mlpackage is a directory, while mlmodel is a file
        if lowercasedPath.hasSuffix(".mlpackage") && isDirectory.boolValue {
          modelURL = possibleURL
        } else if lowercasedPath.hasSuffix(".mlmodel") && !isDirectory.boolValue {
          modelURL = possibleURL
        }
      }
    } else {
      // Check for precompiled models in the bundle
      if let compiledURL = Bundle.main.url(forResource: modelPathOrName, withExtension: "mlmodelc")
      {
        modelURL = compiledURL
      } else if let packageURL = Bundle.main.url(
        forResource: modelPathOrName, withExtension: "mlpackage")
      {
        modelURL = packageURL
      }
    }

    // If the model URL is still unresolved, check Flutter assets
    if modelURL == nil {
      // For absolute paths, use them directly and allow directories
      var isDirectory: ObjCBool = false
      if fileManager.fileExists(atPath: modelPathOrName, isDirectory: &isDirectory) {
        modelURL = URL(fileURLWithPath: modelPathOrName)
      }

      // Handle paths that include folder structure
      if modelPathOrName.contains("/") && modelURL == nil {
        let components = modelPathOrName.components(separatedBy: "/")
        let fileName = components.last ?? ""
        let directory = components.dropLast().joined(separator: "/")
        let assetDirectory = "flutter_assets/\(directory)"

        // Try resolving with the filename as-is
        if let assetPath = Bundle.main.path(
          forResource: fileName, ofType: nil, inDirectory: assetDirectory)
        {
          modelURL = URL(fileURLWithPath: assetPath)
        }

        // Try resolving by separating the extension
        if modelURL == nil && fileName.contains(".") {
          let fileComponents = fileName.components(separatedBy: ".")
          let name = fileComponents.dropLast().joined(separator: ".")
          let ext = fileComponents.last ?? ""

          if let assetPath = Bundle.main.path(
            forResource: name, ofType: ext, inDirectory: assetDirectory)
          {
            modelURL = URL(fileURLWithPath: assetPath)
          }
        }
      }

      // Check the asset directory directly
      if modelURL == nil && modelPathOrName.contains("/") {
        let assetPath = "flutter_assets/\(modelPathOrName)"

        if let directPath = Bundle.main.path(forResource: assetPath, ofType: nil) {
          modelURL = URL(fileURLWithPath: directPath)
        }
      }

      // If there is no folder structure, search by filename only
      if modelURL == nil {
        let fileName = modelPathOrName.components(separatedBy: "/").last ?? modelPathOrName

        // Check the Flutter assets root
        if let assetPath = Bundle.main.path(
          forResource: fileName, ofType: nil, inDirectory: "flutter_assets")
        {
          modelURL = URL(fileURLWithPath: assetPath)
        }

        // Try resolving by separating the extension
        if modelURL == nil && fileName.contains(".") {
          let fileComponents = fileName.components(separatedBy: ".")
          let name = fileComponents.dropLast().joined(separator: ".")
          let ext = fileComponents.last ?? ""

          if let assetPath = Bundle.main.path(
            forResource: name, ofType: ext, inDirectory: "flutter_assets")
          {
            modelURL = URL(fileURLWithPath: assetPath)
          }
        }
      }
    }

    // Check resource bundles (for example, Example/Flutter/App.frameworks/App.framework)
    if modelURL == nil {
      for bundle in Bundle.allBundles {
        // Handle paths that include folder structure
        if modelPathOrName.contains("/") {
          let components = modelPathOrName.components(separatedBy: "/")
          let fileName = components.last ?? ""

          // Search using the filename only
          if let path = bundle.path(forResource: fileName, ofType: nil) {
            modelURL = URL(fileURLWithPath: path)
            break
          }

          // Search by separating the extension
          if fileName.contains(".") {
            let fileComponents = fileName.components(separatedBy: ".")
            let name = fileComponents.dropLast().joined(separator: ".")
            let ext = fileComponents.last ?? ""

            if let path = bundle.path(forResource: name, ofType: ext) {
              modelURL = URL(fileURLWithPath: path)
              break
            }
          }
        }
      }
    }

    guard let unwrappedModelURL = modelURL else {
      NSLog("YOLO: Model not found at path: %@", modelPathOrName)
      completion?(.failure(PredictorError.modelFileNotFound))
      return
    }

    func handleSuccess(predictor: Predictor) {
      self.predictor = predictor
      completion?(.success(self))
    }

    // Common failure handling for all tasks
    func handleFailure(_ error: Error) {
      NSLog("YOLO: Failed to load model: %@", String(describing: error))
      completion?(.failure(error))
    }

    switch task {
    case .classify:
      Classifier.create(unwrappedModelURL: unwrappedModelURL, useGpu: useGpu) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .segment:
      Segmenter.create(
        unwrappedModelURL: unwrappedModelURL, useGpu: useGpu, numItemsThreshold: numItemsThreshold
      ) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .pose:
      PoseEstimater.create(
        unwrappedModelURL: unwrappedModelURL, useGpu: useGpu, numItemsThreshold: numItemsThreshold
      ) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .obb:
      ObbDetector.create(
        unwrappedModelURL: unwrappedModelURL, useGpu: useGpu, numItemsThreshold: numItemsThreshold
      ) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    default:
      ObjectDetector.create(
        unwrappedModelURL: unwrappedModelURL, useGpu: useGpu, numItemsThreshold: numItemsThreshold
      ) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }
    }
  }

  public func callAsFunction(_ uiImage: UIImage, returnAnnotatedImage: Bool = true) -> YOLOResult {
    let ciImage = CIImage(image: uiImage)!
    let result = predictor.predictOnImage(image: ciImage)
    return result
  }

  public func callAsFunction(_ ciImage: CIImage, returnAnnotatedImage: Bool = true) -> YOLOResult {
    var result = predictor.predictOnImage(image: ciImage)
    if returnAnnotatedImage {
      let annotatedImage = drawYOLODetections(on: ciImage, result: result)
      result.annotatedImage = annotatedImage
    }
    return result
  }

  public func callAsFunction(_ cgImage: CGImage, returnAnnotatedImage: Bool = true) -> YOLOResult {
    let ciImage = CIImage(cgImage: cgImage)
    var result = predictor.predictOnImage(image: ciImage)
    if returnAnnotatedImage {
      let annotatedImage = drawYOLODetections(on: ciImage, result: result)
      result.annotatedImage = annotatedImage
    }
    return result
  }

  public func callAsFunction(
    _ resourceName: String,
    withExtension ext: String? = nil,
    returnAnnotatedImage: Bool = true
  ) -> YOLOResult {
    guard let url = Bundle.main.url(forResource: resourceName, withExtension: ext),
      let data = try? Data(contentsOf: url),
      let uiImage = UIImage(data: data)
    else {
      return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }

  public func callAsFunction(
    _ remoteURL: URL?,
    returnAnnotatedImage: Bool = true
  ) -> YOLOResult {
    guard let remoteURL = remoteURL,
      let data = try? Data(contentsOf: remoteURL),
      let uiImage = UIImage(data: data)
    else {
      return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }

  public func callAsFunction(
    _ localPath: String,
    returnAnnotatedImage: Bool = true
  ) -> YOLOResult {
    let fileURL = URL(fileURLWithPath: localPath)
    guard let data = try? Data(contentsOf: fileURL),
      let uiImage = UIImage(data: data)
    else {
      return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }

  @MainActor @available(iOS 16.0, *)
  public func callAsFunction(
    _ swiftUIImage: SwiftUI.Image,
    returnAnnotatedImage: Bool = true
  ) -> YOLOResult {
    let renderer = ImageRenderer(content: swiftUIImage)
    guard let uiImage = renderer.uiImage else {
      return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }
}
