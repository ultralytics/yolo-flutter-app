// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Foundation
import UIKit
import Flutter

/// Manages multiple YOLO instances with unique IDs
@MainActor
class YOLOInstanceManager {
    static let shared = YOLOInstanceManager()
    
    private var instances: [String: YOLO] = [:]
    private var loadingStates: [String: Bool] = [:]
    private var loadCompletionHandlers: [String: [(Result<YOLO, Error>) -> Void]] = [:]
    
    private init() {
        // Initialize default instance for backward compatibility
        createInstance(instanceId: "default")
    }
    
    /// Creates a new YOLO instance with the given ID
    func createInstance(instanceId: String) {
        // Initialize empty handlers for this instance
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
        
        YOLO(resolvedModelPath, task: task) { [weak self] result in
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
        let originalConfThreshold = yolo.confidenceThreshold
        let originalIouThreshold = yolo.iouThreshold
        
        // Apply custom thresholds if provided
        if let confThreshold = confidenceThreshold {
            yolo.confidenceThreshold = confThreshold
        }
        if let iouThres = iouThreshold {
            yolo.iouThreshold = iouThres
        }
        
        result = yolo.callAsFunction(image)
        
        // Restore original thresholds
        yolo.confidenceThreshold = originalConfThreshold
        yolo.iouThreshold = originalIouThreshold
        
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
        
        let fileManager = FileManager.default
        
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
        var flutterResults: [[String: Any]] = []
        
        for box in result.boxes {
            var boxDict: [String: Any] = [
                "cls": box.cls,
                "confidence": box.conf,
                "index": box.index,
            ]
            
            boxDict["x"] = box.xywhn.minX
            boxDict["y"] = box.xywhn.minY
            boxDict["width"] = box.xywhn.width
            boxDict["height"] = box.xywhn.height
            
            boxDict["xImg"] = box.xywh.minX
            boxDict["yImg"] = box.xywh.minY
            boxDict["widthImg"] = box.xywh.width
            boxDict["heightImg"] = box.xywh.height
            
            boxDict["bbox"] = [box.xywh.minX, box.xywh.minY, box.xywh.width, box.xywh.height]
            
            flutterResults.append(boxDict)
        }
        
        var resultDict: [String: Any] = [
            "boxes": flutterResults
        ]
        
        if let annotatedImage = result.annotatedImage {
            if let imageData = annotatedImage.pngData() {
                resultDict["annotatedImage"] = FlutterStandardTypedData(bytes: imageData)
            }
        }
        
        return resultDict
    }
}