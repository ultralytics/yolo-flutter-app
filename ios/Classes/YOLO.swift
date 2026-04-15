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
  public var iouThreshold: Double = 0.4 {
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
    print("YOLO.init: Received modelPath: \(modelPathOrName)")

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
      print("YOLO Debug: Searching for model at path: \(modelPathOrName)")

      // For absolute paths, use them directly and allow directories
      var isDirectory: ObjCBool = false
      if fileManager.fileExists(atPath: modelPathOrName, isDirectory: &isDirectory) {
        print(
          "YOLO Debug: Found model at absolute path: \(modelPathOrName) (isDirectory: \(isDirectory.boolValue))"
        )
        modelURL = URL(fileURLWithPath: modelPathOrName)
      }

      // Handle paths that include folder structure
      if modelPathOrName.contains("/") && modelURL == nil {
        let components = modelPathOrName.components(separatedBy: "/")
        let fileName = components.last ?? ""
        let directory = components.dropLast().joined(separator: "/")
        let assetDirectory = "flutter_assets/\(directory)"

        print("YOLO Debug: Checking in asset directory: \(assetDirectory) for file: \(fileName)")

        // Try resolving with the filename as-is
        if let assetPath = Bundle.main.path(
          forResource: fileName, ofType: nil, inDirectory: assetDirectory)
        {
          print("YOLO Debug: Found model in assets directory: \(assetPath)")
          modelURL = URL(fileURLWithPath: assetPath)
        }

        // Try resolving by separating the extension
        if modelURL == nil && fileName.contains(".") {
          let fileComponents = fileName.components(separatedBy: ".")
          let name = fileComponents.dropLast().joined(separator: ".")
          let ext = fileComponents.last ?? ""

          print(
            "YOLO Debug: Trying with separated name: \(name) and extension: \(ext) in directory: \(assetDirectory)"
          )

          if let assetPath = Bundle.main.path(
            forResource: name, ofType: ext, inDirectory: assetDirectory)
          {
            print("YOLO Debug: Found model with separated extension: \(assetPath)")
            modelURL = URL(fileURLWithPath: assetPath)
          }
        }
      }

      // Check the asset directory directly
      if modelURL == nil && modelPathOrName.contains("/") {
        let assetPath = "flutter_assets/\(modelPathOrName)"
        print("YOLO Debug: Checking direct asset path: \(assetPath)")

        if let directPath = Bundle.main.path(forResource: assetPath, ofType: nil) {
          print("YOLO Debug: Found model at direct asset path: \(directPath)")
          modelURL = URL(fileURLWithPath: directPath)
        }
      }

      // If there is no folder structure, search by filename only
      if modelURL == nil {
        let fileName = modelPathOrName.components(separatedBy: "/").last ?? modelPathOrName
        print("YOLO Debug: Checking filename only: \(fileName) in flutter_assets root")

        // Check the Flutter assets root
        if let assetPath = Bundle.main.path(
          forResource: fileName, ofType: nil, inDirectory: "flutter_assets")
        {
          print("YOLO Debug: Found model in flutter_assets root: \(assetPath)")
          modelURL = URL(fileURLWithPath: assetPath)
        }

        // Try resolving by separating the extension
        if modelURL == nil && fileName.contains(".") {
          let fileComponents = fileName.components(separatedBy: ".")
          let name = fileComponents.dropLast().joined(separator: ".")
          let ext = fileComponents.last ?? ""

          print("YOLO Debug: Trying with separated filename: \(name) and extension: \(ext)")

          if let assetPath = Bundle.main.path(
            forResource: name, ofType: ext, inDirectory: "flutter_assets")
          {
            print(
              "YOLO Debug: Found model with separated extension in flutter_assets: \(assetPath)")
            modelURL = URL(fileURLWithPath: assetPath)
          }
        }
      }
    }

    // Check resource bundles (for example, Example/Flutter/App.frameworks/App.framework)
    if modelURL == nil {
      for bundle in Bundle.allBundles {
        let bundleID = bundle.bundleIdentifier ?? "unknown"
        print("YOLO Debug: Checking bundle: \(bundleID)")

        // Handle paths that include folder structure
        if modelPathOrName.contains("/") {
          let components = modelPathOrName.components(separatedBy: "/")
          let fileName = components.last ?? ""

          // Search using the filename only
          if let path = bundle.path(forResource: fileName, ofType: nil) {
            print("YOLO Debug: Found model in bundle \(bundleID): \(path)")
            modelURL = URL(fileURLWithPath: path)
            break
          }

          // Search by separating the extension
          if fileName.contains(".") {
            let fileComponents = fileName.components(separatedBy: ".")
            let name = fileComponents.dropLast().joined(separator: ".")
            let ext = fileComponents.last ?? ""

            if let path = bundle.path(forResource: name, ofType: ext) {
              print("YOLO Debug: Found model with ext in bundle \(bundleID): \(path)")
              modelURL = URL(fileURLWithPath: path)
              break
            }
          }
        }
      }
    }

    guard let unwrappedModelURL = modelURL else {
      print("YOLO Error: Model not found at path: \(modelPathOrName)")
      print("YOLO Debug: Original model path: \(modelPathOrName)")
      print("YOLO Debug: Lowercased path: \(lowercasedPath)")

      // Check if the path exists as directory
      var isDirectory: ObjCBool = false
      if fileManager.fileExists(atPath: modelPathOrName, isDirectory: &isDirectory) {
        print("YOLO Debug: Path exists. Is directory: \(isDirectory.boolValue)")

        // If it's a directory and ends with .mlpackage, it should have been found
        if isDirectory.boolValue && lowercasedPath.hasSuffix(".mlpackage") {
          print("YOLO Error: mlpackage directory exists but was not properly recognized")
        }
      } else {
        print("YOLO Debug: Path does not exist")
      }

      // Print the list of available bundles and assets
      print("YOLO Debug: Available bundles:")
      for bundle in Bundle.allBundles {
        print(" - \(bundle.bundleIdentifier ?? "unknown"): \(bundle.bundlePath)")
      }
      print("YOLO Debug: Checking if flutter_assets directory exists:")
      let flutterAssetsPath = Bundle.main.bundlePath + "/flutter_assets"
      if fileManager.fileExists(atPath: flutterAssetsPath) {
        print(" - flutter_assets exists at: \(flutterAssetsPath)")
        // List files inside flutter_assets
        do {
          let items = try fileManager.contentsOfDirectory(atPath: flutterAssetsPath)
          print("YOLO Debug: Files in flutter_assets:")
          for item in items {
            print(" - \(item)")
          }
        } catch {
          print("YOLO Debug: Error listing flutter_assets: \(error)")
        }
      } else {
        print(" - flutter_assets NOT found")
      }

      completion?(.failure(PredictorError.modelFileNotFound))
      return
    }

    func handleSuccess(predictor: Predictor) {
      self.predictor = predictor
      completion?(.success(self))
    }

    // Common failure handling for all tasks
    func handleFailure(_ error: Error) {
      print("Failed to load model with error: \(error)")
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
    var result = predictor.predictOnImage(image: ciImage)
    //        if returnAnnotatedImage {
    //            let annotatedImage = drawYOLODetections(on: ciImage, result: result)
    //            result.annotatedImage = annotatedImage
    //        }
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
