// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Flutter
import UIKit

/// Class that manages YOLO models as a singleton instance
@MainActor
class SingleImageYOLO {
  static let shared = SingleImageYOLO()
  private var yolo: YOLO?
  private var isLoadingModel = false
  private var loadCompletionHandlers: [(Result<YOLO, Error>) -> Void] = []

  private init() {}

  func loadModel(
    modelName: String, task: YOLOTask, completion: @escaping (Result<Void, Error>) -> Void
  ) {
    // モデルが既に読み込まれている場合は成功を返す
    if yolo != nil {
      completion(.success(()))
      return
    }

    // モデルが読み込み中の場合は完了ハンドラーを追加
    if isLoadingModel {
      loadCompletionHandlers.append({ result in
        switch result {
        case .success:
          completion(.success(()))
        case .failure(let error):
          completion(.failure(error))
        }
      })
      return
    }

    isLoadingModel = true

    // Flutterアセットの処理
    let resolvedModelPath = resolveModelPath(modelName)

    // YOLOモデルを初期化し読み込む
    YOLO(resolvedModelPath, task: task) { [weak self] result in
      guard let self = self else { return }

      self.isLoadingModel = false

      switch result {
      case .success(let loadedYolo):
        self.yolo = loadedYolo
        completion(.success(()))

        // 保留中の完了ハンドラーを実行
        for handler in self.loadCompletionHandlers {
          handler(.success(loadedYolo))
        }

      case .failure(let error):
        completion(.failure(error))

        // 保留中の完了ハンドラーにエラーを通知
        for handler in self.loadCompletionHandlers {
          handler(.failure(error))
        }
      }

      self.loadCompletionHandlers.removeAll()
    }
  }

  // モデルパスを解決するヘルパーメソッド
  private func resolveModelPath(_ modelPath: String) -> String {
    // 既に絶対パスの場合はそのまま返す
    if modelPath.hasPrefix("/") {
      return modelPath
    }

    let fileManager = FileManager.default

    // Flutterアセットからのパス解決（例：assets/models/yolo11n.mlmodel）
    if modelPath.contains("/") {
      let components = modelPath.components(separatedBy: "/")
      let fileName = components.last ?? ""
      let fileNameWithoutExt = fileName.components(separatedBy: ".").first ?? fileName
      let directory = components.dropLast().joined(separator: "/")

      // 検索パスのリスト
      let searchPaths = [
        "flutter_assets/\(modelPath)",  // 完全なパス (assets/models/yolo11n.mlmodel)
        "flutter_assets/\(directory)",  // ディレクトリのみ (assets/models)
        "flutter_assets",  // Flutterアセットのルート
        "",  // バンドルのルート
      ]

      // 各検索パスでファイルを探す
      for searchPath in searchPaths {
        // 完全な名前で検索
        if !searchPath.isEmpty,
          let assetPath = Bundle.main.path(
            forResource: fileName, ofType: nil, inDirectory: searchPath)
        {
          return assetPath
        }

        // 名前と拡張子で検索
        if fileName.contains(".") {
          let fileComponents = fileName.components(separatedBy: ".")
          let name = fileComponents.dropLast().joined(separator: ".")
          let ext = fileComponents.last ?? ""

          if !searchPath.isEmpty,
            let assetPath = Bundle.main.path(
              forResource: name, ofType: ext, inDirectory: searchPath)
          {
            return assetPath
          }
        }

        // ファイル名だけで検索
        if !searchPath.isEmpty,
          let assetPath = Bundle.main.path(
            forResource: fileNameWithoutExt, ofType: nil, inDirectory: searchPath)
        {
          return assetPath
        }
      }

      // 全バンドル内を検索
      for bundle in Bundle.allBundles {
        let bundleID = bundle.bundleIdentifier ?? "unknown"

        // 完全な名前で検索
        if let assetPath = bundle.path(forResource: fileName, ofType: nil) {
          return assetPath
        }

        // 名前と拡張子で検索
        if fileName.contains(".") {
          let fileComponents = fileName.components(separatedBy: ".")
          let name = fileComponents.dropLast().joined(separator: ".")
          let ext = fileComponents.last ?? ""

          if let assetPath = bundle.path(forResource: name, ofType: ext) {
            return assetPath
          }
        }

        // ファイル名だけで検索
        if let assetPath = bundle.path(forResource: fileNameWithoutExt, ofType: nil) {
          return assetPath
        }
      }

      // ファイルが見つからなかった場合はファイルシステムに直接アクセスする
      let possiblePaths = [
        Bundle.main.bundlePath + "/flutter_assets/\(modelPath)",
        Bundle.main.bundlePath + "/flutter_assets/\(fileName)",
      ]

      for path in possiblePaths {
        if fileManager.fileExists(atPath: path) {
          return path
        }
      }
    } else {
      // モデルパスがファイル名のみの場合
      // すべてのバンドルを検索
      for bundle in Bundle.allBundles {
        let bundleID = bundle.bundleIdentifier ?? "unknown"

        if let path = bundle.path(forResource: modelPath, ofType: nil) {
          return path
        }

        // 名前と拡張子で検索
        if modelPath.contains(".") {
          let fileComponents = modelPath.components(separatedBy: ".")
          let name = fileComponents.dropLast().joined(separator: ".")
          let ext = fileComponents.last ?? ""

          if let path = bundle.path(forResource: name, ofType: ext) {
            return path
          }
        }
      }

      // Flutterアセットで検索
      if let path = Bundle.main.path(
        forResource: modelPath, ofType: nil, inDirectory: "flutter_assets")
      {
        return path
      }
    }

    // ファイルが見つからなかった場合、元のパスをそのまま返す
    return modelPath
  }

  func predict(imageData: Data) -> [String: Any]? {
    guard let yolo = self.yolo, let uiImage = UIImage(data: imageData) else {
      return nil
    }

    // 推論を実行
    let result = yolo(uiImage)

    // YOLOResultをFlutter用のディクショナリに変換
    return convertToFlutterFormat(result: result)
  }

  private func convertToFlutterFormat(result: YOLOResult) -> [String: Any] {
    // 検出結果を変換
    var flutterResults: [[String: Any]] = []

    for box in result.boxes {
      var boxDict: [String: Any] = [
        "cls": box.cls,
        "confidence": box.conf,
        "index": box.index,
      ]

      // 正規化された座標を追加
      boxDict["x"] = box.xywhn.minX
      boxDict["y"] = box.xywhn.minY
      boxDict["width"] = box.xywhn.width
      boxDict["height"] = box.xywhn.height

      // 画像座標値（ピクセル単位）も追加
      boxDict["xImg"] = box.xywh.minX
      boxDict["yImg"] = box.xywh.minY
      boxDict["widthImg"] = box.xywh.width
      boxDict["heightImg"] = box.xywh.height

      // バウンディングボックス座標をリスト形式でも追加
      boxDict["bbox"] = [box.xywh.minX, box.xywh.minY, box.xywh.width, box.xywh.height]

      flutterResults.append(boxDict)
    }

    // 結果全体を格納するディクショナリ
    var resultDict: [String: Any] = [
      "boxes": flutterResults
    ]

    // アノテーション画像がある場合、それをBase64エンコードして追加
    if let annotatedImage = result.annotatedImage {
      if let imageData = annotatedImage.pngData() {
        resultDict["annotatedImage"] = FlutterStandardTypedData(bytes: imageData)
      }
    }

    return resultDict
  }
}

@MainActor
public class YoloPlugin: NSObject, FlutterPlugin {
  public static func register(with registrar: FlutterPluginRegistrar) {
    // 1) Register the platform view
    let factory = SwiftYoloPlatformViewFactory(messenger: registrar.messenger())
    registrar.register(factory, withId: "com.ultralytics.yolo/YoloPlatformView")

    // 2) Register the method channel for single-image inference
    let channel = FlutterMethodChannel(
      name: "yolo_single_image_channel",
      binaryMessenger: registrar.messenger()
    )
    let instance = YoloPlugin()
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  // モデルが存在するかどうかを確認する関数
  private func checkModelExists(modelPath: String) -> [String: Any] {
    let fileManager = FileManager.default
    var resultMap: [String: Any] = [
      "exists": false,
      "path": modelPath,
      "location": "unknown",
    ]

    // モデルパスの解決
    let lowercasedPath = modelPath.lowercased()

    // 絶対パスのチェック
    if modelPath.hasPrefix("/") {
      if fileManager.fileExists(atPath: modelPath) {
        resultMap["exists"] = true
        resultMap["location"] = "file_system"
        resultMap["absolutePath"] = modelPath
        return resultMap
      }
    }

    // Flutterアセットのパス解決（複数階層）
    if modelPath.contains("/") {
      let components = modelPath.components(separatedBy: "/")
      let fileName = components.last ?? ""
      let directory = components.dropLast().joined(separator: "/")

      // 指定されたディレクトリ内のファイルをチェック
      let assetPath = "flutter_assets/\(directory)"
      if let fullPath = Bundle.main.path(forResource: fileName, ofType: nil, inDirectory: assetPath)
      {
        resultMap["exists"] = true
        resultMap["location"] = "flutter_assets_directory"
        resultMap["absolutePath"] = fullPath
        return resultMap
      }

      // 拡張子分割を試みる（例：yolo11n.mlmodel -> yolo11n, mlmodel）
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

    // Flutterアセットルートでのチェック
    let fileName = modelPath.components(separatedBy: "/").last ?? modelPath
    if let fullPath = Bundle.main.path(
      forResource: fileName, ofType: nil, inDirectory: "flutter_assets")
    {
      resultMap["exists"] = true
      resultMap["location"] = "flutter_assets_root"
      resultMap["absolutePath"] = fullPath
      return resultMap
    }

    // バンドル内のファイル名のみで検索
    // 拡張子分割
    let fileComponents = fileName.components(separatedBy: ".")
    if fileComponents.count > 1 {
      let name = fileComponents.dropLast().joined(separator: ".")
      let ext = fileComponents.last ?? ""

      // 通常のバンドルリソース
      if let fullPath = Bundle.main.path(forResource: name, ofType: ext) {
        resultMap["exists"] = true
        resultMap["location"] = "bundle_resource"
        resultMap["absolutePath"] = fullPath
        return resultMap
      }
    }

    // バンドル内のコンパイル済みモデルをチェック
    if let compiledURL = Bundle.main.url(forResource: fileName, withExtension: "mlmodelc") {
      resultMap["exists"] = true
      resultMap["location"] = "bundle_compiled"
      resultMap["absolutePath"] = compiledURL.path
      return resultMap
    }

    // バンドル内のMLPackageをチェック
    if let packageURL = Bundle.main.url(forResource: fileName, withExtension: "mlpackage") {
      resultMap["exists"] = true
      resultMap["location"] = "bundle_package"
      resultMap["absolutePath"] = packageURL.path
      return resultMap
    }

    return resultMap
  }

  // ストレージパスを取得する関数
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

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    Task { @MainActor in
      switch call.method {
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

        do {
          try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<Void, Error>) in
            SingleImageYOLO.shared.loadModel(modelName: modelPath, task: task) { modelResult in
              switch modelResult {
              case .success:
                continuation.resume()
              case .failure(let error):
                continuation.resume(throwing: error)
              }
            }
          }
          result(nil)  // 成功
        } catch {
          result(
            FlutterError(
              code: "model_load_error", message: error.localizedDescription, details: nil))
        }

      case "predictSingleImage":
        guard let args = call.arguments as? [String: Any],
          let data = args["image"] as? FlutterStandardTypedData
        else {
          result(
            FlutterError(
              code: "bad_args", message: "Invalid arguments for predictSingleImage", details: nil))
          return
        }

        // 実際に画像推論を実行
        if let resultDict = SingleImageYOLO.shared.predict(imageData: data.data) {
          result(resultDict)
        } else {
          result(
            FlutterError(code: "inference_error", message: "Failed to run inference", details: nil))
        }

      case "checkModelExists":
        guard let args = call.arguments as? [String: Any],
          let modelPath = args["modelPath"] as? String
        else {
          result(
            FlutterError(
              code: "bad_args", message: "Invalid arguments for checkModelExists", details: nil))
          return
        }

        let checkResult = checkModelExists(modelPath: modelPath)
        result(checkResult)

      case "getStoragePaths":
        let paths = getStoragePaths()
        result(paths)

      case "setModel":
        guard let args = call.arguments as? [String: Any],
          let viewId = args["viewId"] as? Int,
          let modelPath = args["modelPath"] as? String,
          let taskString = args["task"] as? String
        else {
          result(
            FlutterError(code: "bad_args", message: "Invalid arguments for setModel", details: nil)
          )
          return
        }

        let task = YOLOTask.fromString(taskString)

        // Get the YoloView instance from the factory
        if let yoloView = SwiftYoloPlatformViewFactory.getYoloView(for: viewId) {
          yoloView.setModel(modelPathOrName: modelPath, task: task) { modelResult in
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
              message: "YoloView with id \(viewId) not found",
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
