// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML
import Flutter
import UIKit

@MainActor
public class YOLOPlugin: NSObject, FlutterPlugin {
  // Dictionary to store channels for each instance
  private static var instanceChannels: [String: FlutterMethodChannel] = [:]
  // Store the registrar for creating new channels
  private static var pluginRegistrar: FlutterPluginRegistrar?

  public static func register(with registrar: FlutterPluginRegistrar) {
    // Store the registrar for later use
    pluginRegistrar = registrar
    // 1) Register the platform view
    let factory = SwiftYOLOPlatformViewFactory(messenger: registrar.messenger())
    registrar.register(factory, withId: "com.ultralytics.yolo/YOLOPlatformView")

    // 2) Register the default method channel for backward compatibility
    let defaultChannel = FlutterMethodChannel(
      name: "yolo_single_image_channel",
      binaryMessenger: registrar.messenger()
    )
    let instance = YOLOPlugin()
    registrar.addMethodCallDelegate(instance, channel: defaultChannel)
  }

  private func registerInstanceChannel(instanceId: String, messenger: FlutterBinaryMessenger) {
    let channelName = "yolo_single_image_channel_\(instanceId)"
    let channel = FlutterMethodChannel(name: channelName, binaryMessenger: messenger)
    let instance = YOLOPlugin()
    // Store the channel for later use
    YOLOPlugin.instanceChannels[instanceId] = channel
    // Register this instance as the method call delegate
    if let registrar = YOLOPlugin.pluginRegistrar {
      registrar.addMethodCallDelegate(instance, channel: channel)
    }
  }

  private func checkModelExists(modelPath: String) -> [String: Any] {
    let fileManager = FileManager.default
    var resultMap: [String: Any] = [
      "exists": false,
      "path": modelPath,
      "location": "unknown",
    ]

    let lowercasedPath = modelPath.lowercased()

    if modelPath.hasPrefix("/") {
      if fileManager.fileExists(atPath: modelPath) {
        resultMap["exists"] = true
        resultMap["location"] = "file_system"
        resultMap["absolutePath"] = modelPath
        return resultMap
      }
    }

    if modelPath.contains("/") {
      let components = modelPath.components(separatedBy: "/")
      let fileName = components.last ?? ""
      let directory = components.dropLast().joined(separator: "/")

      let assetPath = "flutter_assets/\(directory)"
      if let fullPath = Bundle.main.path(forResource: fileName, ofType: nil, inDirectory: assetPath)
      {
        resultMap["exists"] = true
        resultMap["location"] = "flutter_assets_directory"
        resultMap["absolutePath"] = fullPath
        return resultMap
      }

      let fileComponents = fileName.components(separatedBy: ".")
      if fileComponents.count > 1 {
        let name = fileComponents.dropLast().joined(separator: ".")
        let ext = fileComponents.last ?? ""

        if let fullPath = Bundle.main.path(forResource: name, ofType: ext, inDirectory: assetPath) {
          resultMap["exists"] = true
          resultMap["location"] = "flutter_assets_directory_with_ext"
          resultMap["absolutePath"] = fullPath
          return resultMap
        }
      }
    }

    let fileName = modelPath.components(separatedBy: "/").last ?? modelPath
    if let fullPath = Bundle.main.path(
      forResource: fileName, ofType: nil, inDirectory: "flutter_assets")
    {
      resultMap["exists"] = true
      resultMap["location"] = "flutter_assets_root"
      resultMap["absolutePath"] = fullPath
      return resultMap
    }

    let fileComponents = fileName.components(separatedBy: ".")
    if fileComponents.count > 1 {
      let name = fileComponents.dropLast().joined(separator: ".")
      let ext = fileComponents.last ?? ""

      if let fullPath = Bundle.main.path(forResource: name, ofType: ext) {
        resultMap["exists"] = true
        resultMap["location"] = "bundle_resource"
        resultMap["absolutePath"] = fullPath
        return resultMap
      }
    }

    if let compiledURL = Bundle.main.url(forResource: fileName, withExtension: "mlmodelc") {
      resultMap["exists"] = true
      resultMap["location"] = "bundle_compiled"
      resultMap["absolutePath"] = compiledURL.path
      return resultMap
    }

    if let packageURL = Bundle.main.url(forResource: fileName, withExtension: "mlpackage") {
      resultMap["exists"] = true
      resultMap["location"] = "bundle_package"
      resultMap["absolutePath"] = packageURL.path
      return resultMap
    }

    return resultMap
  }

  private func getStoragePaths() -> [String: String?] {
    let fileManager = FileManager.default
    let documentsDirectory = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first
    let cachesDirectory = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first
    let applicationSupportDirectory = fileManager.urls(
      for: .applicationSupportDirectory, in: .userDomainMask
    ).first

    return [
      "internal": applicationSupportDirectory?.path,
      "cache": cachesDirectory?.path,
      "documents": documentsDirectory?.path,
    ]
  }

  private func inspectModel(modelPath: String) throws -> [String: Any] {
    let checkResult = checkModelExists(modelPath: modelPath)
    let resolvedPath = (checkResult["absolutePath"] as? String) ?? modelPath
    let url = URL(fileURLWithPath: resolvedPath)
    let ext = url.pathExtension.lowercased()

    let model: MLModel
    if ext == "mlmodelc" {
      model = try MLModel(contentsOf: url)
    } else {
      let compiledURL = try MLModel.compileModel(at: url)
      model = try MLModel(contentsOf: compiledURL)
    }

    let creatorDefined =
      model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String]
      ?? [:]
    let labels = parseLabels(from: creatorDefined)

    var result: [String: Any] = [
      "path": resolvedPath,
      "task": creatorDefined["task"] ?? "",
      "labels": labels,
    ]

    if let names = creatorDefined["names"] {
      result["names"] = names
    }
    if let description = creatorDefined["description"] {
      result["description"] = description
    }
    if let imgsz = creatorDefined["imgsz"] {
      result["imgsz"] = imgsz
    }
    if let stride = creatorDefined["stride"] {
      result["stride"] = stride
    }
    if let channels = creatorDefined["channels"] {
      result["channels"] = channels
    }
    if let end2end = creatorDefined["end2end"] {
      result["end2end"] = end2end
    }

    return result
  }

  private func parseLabels(from userDefined: [String: String]) -> [String] {
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

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    Task { @MainActor in
      switch call.method {
      case "createInstance":
        guard let args = call.arguments as? [String: Any],
          let instanceId = args["instanceId"] as? String
        else {
          result(
            FlutterError(
              code: "bad_args", message: "Invalid arguments for createInstance", details: nil)
          )
          return
        }

        YOLOInstanceManager.shared.createInstance(instanceId: instanceId)

        // Register a new channel for this instance
        if let registrar = YOLOPlugin.pluginRegistrar {
          registerInstanceChannel(instanceId: instanceId, messenger: registrar.messenger())
        }

        result(nil)

      case "loadModel":
        guard let args = call.arguments as? [String: Any],
          let modelPath = args["modelPath"] as? String,
          let taskString = args["task"] as? String
        else {
          result(
            FlutterError(code: "bad_args", message: "Invalid arguments for loadModel", details: nil)
          )
          return
        }

        let task = YOLOTask.fromString(taskString)
        let instanceId = args["instanceId"] as? String ?? "default"
        let useGpu = args["useGpu"] as? Bool ?? true
        let numItemsThreshold = args["numItemsThreshold"] as? Int ?? 30

        do {
          try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<Void, Error>) in
            YOLOInstanceManager.shared.loadModel(
              instanceId: instanceId,
              modelName: modelPath,
              task: task,
              useGpu: useGpu,
              numItemsThreshold: numItemsThreshold
            ) { modelResult in
              switch modelResult {
              case .success:
                continuation.resume()
              case .failure(let error):
                continuation.resume(throwing: error)
              }
            }
          }
          result(true)
        } catch {
          result(
            FlutterError(
              code: "MODEL_NOT_FOUND",
              message: error.localizedDescription,
              details: nil
            )
          )
        }

      case "predictSingleImage":
        guard let args = call.arguments as? [String: Any],
          let data = args["image"] as? FlutterStandardTypedData
        else {
          result(
            FlutterError(
              code: "bad_args", message: "Invalid arguments for predictSingleImage", details: nil)
          )
          return
        }

        let instanceId = args["instanceId"] as? String ?? "default"
        let confidenceThreshold = args["confidenceThreshold"] as? Double
        let iouThreshold = args["iouThreshold"] as? Double

        if let resultDict = YOLOInstanceManager.shared.predict(
          instanceId: instanceId,
          imageData: data.data,
          confidenceThreshold: confidenceThreshold,
          iouThreshold: iouThreshold
        ) {
          result(resultDict)
        } else {
          result(
            FlutterError(
              code: "MODEL_NOT_LOADED",
              message: "Model has not been loaded. Call loadModel() first.",
              details: nil
            )
          )
        }

      case "disposeInstance":
        guard let args = call.arguments as? [String: Any],
          let instanceId = args["instanceId"] as? String
        else {
          result(
            FlutterError(
              code: "bad_args", message: "Invalid arguments for disposeInstance", details: nil)
          )
          return
        }

        YOLOInstanceManager.shared.removeInstance(instanceId: instanceId)

        // Remove the channel for this instance
        YOLOPlugin.instanceChannels.removeValue(forKey: instanceId)

        result(nil)

      case "predictorInstance":
        // do nothing with ios
        result(nil)

      case "checkModelExists":
        guard let args = call.arguments as? [String: Any],
          let modelPath = args["modelPath"] as? String
        else {
          result(
            FlutterError(
              code: "bad_args", message: "Invalid arguments for checkModelExists", details: nil)
          )
          return
        }

        let checkResult = checkModelExists(modelPath: modelPath)
        result(checkResult)

      case "getStoragePaths":
        let paths = getStoragePaths()
        result(paths)

      case "inspectModel":
        guard let args = call.arguments as? [String: Any],
          let modelPath = args["modelPath"] as? String
        else {
          result(
            FlutterError(
              code: "bad_args", message: "Invalid arguments for inspectModel", details: nil)
          )
          return
        }

        do {
          result(try inspectModel(modelPath: modelPath))
        } catch {
          result(
            FlutterError(
              code: "MODEL_INSPECTION_FAILED",
              message: error.localizedDescription,
              details: nil
            )
          )
        }

      case "setModel":
        guard let args = call.arguments as? [String: Any],
          let modelPath = args["modelPath"] as? String,
          let taskString = args["task"] as? String
        else {
          result(
            FlutterError(code: "bad_args", message: "Invalid arguments for setModel", details: nil)
          )
          return
        }

        let task = YOLOTask.fromString(taskString)
        let useGpu = args["useGpu"] as? Bool ?? true

        // Handle both String viewId (from Flutter) and Int viewId
        var viewIdInt: Int?
        var instanceId: String = "default"

        if let flutterViewId = args["viewId"] as? String {
          // This is the Flutter string viewId
          instanceId = flutterViewId
          // Try to extract numeric viewId from YOLOView registration
          // This is a workaround - ideally we should store the mapping
          viewIdInt = nil  // We'll search for it below
        } else if let numericViewId = args["viewId"] as? Int {
          viewIdInt = numericViewId
          instanceId = String(numericViewId)
        }

        // Remove existing instance before loading new model to prevent memory leaks
        YOLOInstanceManager.shared.removeInstance(instanceId: instanceId)

        YOLOInstanceManager.shared.createInstance(instanceId: instanceId)

        // Try to find the YOLOView - if we don't have viewIdInt, we need to search
        if viewIdInt == nil {
          // This is not ideal, but for now we'll load the model without the view
          // The view will connect to the model when it's created
          result(
            FlutterError(
              code: "NOT_IMPLEMENTED",
              message: "Model switching with string viewId not fully implemented",
              details: "Please use the default YOLO instance for now"
            )
          )
          return
        }

        // Get the YOLOView instance from the factory
        if let yoloView = SwiftYOLOPlatformViewFactory.getYOLOView(for: viewIdInt!) {
          yoloView.setModel(modelPathOrName: modelPath, task: task, useGpu: useGpu) { modelResult in
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
              code: "VIEW_NOT_FOUND",
              message: "YOLOView with id \(viewIdInt ?? -1) not found",
              details: nil
            )
          )
        }

      default:
        result(FlutterMethodNotImplemented)
      }
    }
  }
}
