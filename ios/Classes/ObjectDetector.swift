import Foundation
import Vision
import UIKit

public class ObjectDetector: Predictor {
    private var detector: VNCoreMLModel!
    private var visionRequest: VNCoreMLRequest?
    private var currentBuffer: CVPixelBuffer?
    private var currentOnResultsListener: ResultsListener?
    private var currentOnInferenceTimeListener: InferenceTimeListener?
    private var currentOnFpsRateListener: FpsRateListener?
    private var screenSize: CGSize?
    private var labels = [String]()
    var t0 = 0.0  // inference start
    var t1 = 0.0  // inference dt
    var t2 = 0.0  // inference dt smoothed
    var t3 = CACurrentMediaTime()  // FPS start
    var t4 = 0.0  // FPS dt smoothed
    
    public init? (yoloModel: any YoloModel) async throws {
        if(yoloModel.task != "detect") {
            throw PredictorError.invalidTask
        }
        
        guard let mlModel = try await yoloModel.loadModel() as? MLModel
        else { return }
        
        guard let userDefined = mlModel.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String]
        else { return }
        
        var allLabels: String = ""
        if let labelsData = userDefined["classes"] {
            allLabels = labelsData
            labels = allLabels.components(separatedBy: ",")
        } else if let labelsData = userDefined["names"] {
            // Remove curly braces and spaces from the input string
            let cleanedInput = labelsData.replacingOccurrences(of: "{", with: "")
                .replacingOccurrences(of: "}", with: "")
                .replacingOccurrences(of: " ", with: "")
            
            // Split the cleaned string into an array of key-value pairs
            let keyValuePairs = cleanedInput.components(separatedBy: ",")


            for pair in keyValuePairs {
                // Split each key-value pair into key and value
                let components = pair.components(separatedBy: ":")
                
                
                // Check if we have at least two components
                if components.count >= 2 {
                    // Get the second component and trim any leading/trailing whitespace
                    let extractedString = components[1].trimmingCharacters(in: .whitespaces)

                    // Remove single quotes if they exist
                    let cleanedString = extractedString.replacingOccurrences(of: "'", with: "")

                    labels.append(cleanedString)
                } else {
                    print("Invalid input string")
                }
            }

        } else {
            fatalError("Invalid metadata format")
        }
        
        
        let bounds: CGRect = await UIScreen.main.bounds
        screenSize = CGSize(width: bounds.width, height: bounds.height)
            
        detector = try! VNCoreMLModel(for: mlModel)
        detector.featureProvider = ThresholdProvider()

        visionRequest = {
            let request = VNCoreMLRequest(model: detector, completionHandler: {
                [weak self] request, error in
                self?.processObservations(for: request, error: error)
            })
            request.imageCropAndScaleOption = .scaleFill
            return request
        }()
    }
    
    public func predict(sampleBuffer: CMSampleBuffer, onResultsListener: ResultsListener?, onInferenceTime: InferenceTimeListener?, onFpsRate: FpsRateListener?) {
        if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            currentBuffer = pixelBuffer
            currentOnResultsListener = onResultsListener
            currentOnInferenceTimeListener = onInferenceTime
            currentOnFpsRateListener = onFpsRate
            
            /// - Tag: MappingOrientation
            // The frame is always oriented based on the camera sensor,
            // so in most cases Vision needs to rotate it for the model to work as expected.
            let imageOrientation: CGImagePropertyOrientation
            switch UIDevice.current.orientation {
            case .portrait:
                imageOrientation = .up
            case .portraitUpsideDown:
                imageOrientation = .down
            case .landscapeLeft:
                imageOrientation = .left
            case .landscapeRight:
                imageOrientation = .right
            case .unknown:
                print("The device orientation is unknown, the predictions may be affected")
                fallthrough
            default:
                imageOrientation = .up
            }
            
            // Invoke a VNRequestHandler with that image
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
            t0 = CACurrentMediaTime()  // inference start
            do {
                if(visionRequest != nil){
                    try handler.perform([visionRequest!])
                }
            } catch {
                print(error)
            }
            t1 = CACurrentMediaTime() - t0  // inference dt

            
            currentBuffer = nil
        }
    }
    
    private var confidenceThreshold = 0.2
    public func setConfidenceThreshold(confidence: Double) {
        confidenceThreshold = confidence
        detector.featureProvider = ThresholdProvider(iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
    }
    
    private var iouThreshold = 0.4
    public func setIouThreshold(iou: Double){
        iouThreshold = iou
        detector.featureProvider = ThresholdProvider(iouThreshold: iouThreshold, confidenceThreshold: confidenceThreshold)
    }
    
    private var numItemsThreshold = 30
    public func setNumItemsThreshold(numItems: Int){
        numItemsThreshold = numItems
    }
    
    private func processObservations(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            if let results = request.results as? [VNRecognizedObjectObservation] {
                var recognitions: [[String:Any]] = []
                
                let width = self.screenSize?.width ?? 375  // 375 pix
                let height = self.screenSize?.height ?? 816  // 812 pix
                let ratio: CGFloat = (height / width) / (4.0 / 3.0)  // .photo

                for i in 0..<100 {
                    if i < results.count && i < self.numItemsThreshold {
                        let prediction = results[i]
                        
                        var rect = prediction.boundingBox  // normalized xywh, origin lower left
                        switch UIDevice.current.orientation {
                        case .portraitUpsideDown:
                            rect = CGRect(x: 1.0 - rect.origin.x - rect.width,
                                          y: 1.0 - rect.origin.y - rect.height,
                                          width: rect.width,
                                          height: rect.height)
                        case .landscapeLeft:
                            rect = CGRect(x: rect.origin.y,
                                          y: 1.0 - rect.origin.x - rect.width,
                                          width: rect.height,
                                          height: rect.width)
                        case .landscapeRight:
                            rect = CGRect(x: 1.0 - rect.origin.y - rect.height,
                                          y: rect.origin.x,
                                          width: rect.height,
                                          height: rect.width)
                        case .unknown:
                            print("The device orientation is unknown, the predictions may be affected")
                            fallthrough
                        default: break
                        }
                        
                        if ratio >= 1 { // iPhone ratio = 1.218
                            let offset = (1 - ratio) * (0.5 - rect.minX)
                            let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
                            rect = rect.applying(transform)
                            rect.size.width *= ratio
                        } else { // iPad ratio = 0.75
                            let offset = (ratio - 1) * (0.5 - rect.maxY)
                            let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
                            rect = rect.applying(transform)
                            rect.size.height /= ratio
                        }

                        // Scale normalized to pixels [375, 812] [width, height]
                        rect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))
                        
                        // The labels array is a list of VNClassificationObservation objects,
                        // with the highest scoring class first in the list.
                        let label = prediction.labels[0].identifier
                        let index = self.labels.firstIndex(of: label) ?? 0
                        let confidence = prediction.labels[0].confidence
                        recognitions.append(["label": label,
                                             "confidence": confidence,
                                             "index": index,
                                             "x": rect.origin.x,
                                             "y": rect.origin.y,
                                             "width": rect.size.width,
                                             "height": rect.size.height])
                    }
                }
                
                self.currentOnResultsListener?.on(predictions: recognitions)
                
                // Measure FPS
                if self.t1 < 10.0 {  // valid dt
                    self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
                }
                self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
                self.t3 = CACurrentMediaTime()

                self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000)  // t2 seconds to ms
                self.currentOnFpsRateListener?.on(fpsRate: 1 / self.t4)
            }
        }
    }
    
    public func predictOnImage(image: CIImage, completion: ([[String:Any]]) -> Void) {
        let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
        let request = VNCoreMLRequest(model: detector)
        var recognitions: [[String:Any]] = []
        
        let screenWidth = self.screenSize?.width ?? 393
        let screenHeight = self.screenSize?.height ?? 852
        let imageWidth = image.extent.width
        let imageHeight = image.extent.height
        let scaleFactor = screenWidth / imageWidth
        let newHeight = imageHeight * scaleFactor
        let screenRatio: CGFloat = (screenHeight / screenWidth) / (4.0 / 3.0)  // .photo
        
        do {
            try requestHandler.perform([request])
            if let results = request.results as? [VNRecognizedObjectObservation] {
                for i in 0..<100 {
                    if i < results.count && i < self.numItemsThreshold {
                        let prediction = results[i]
                        
                        var rect = prediction.boundingBox  // normalized xywh, origin lower left
                        print("rect: \(rect)")
                        
                        if screenRatio >= 1 { // iPhone ratio = 1.218
                            let offset = (1 - screenRatio) * (0.5 - rect.minX)
                            let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
                            rect = rect.applying(transform)
    //                        rect.size.width *= screenRatio
                        } else { // iPad ratio = 0.75
                            let offset = (screenRatio - 1) * (0.5 - rect.maxY)
                            let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
                            rect = rect.applying(transform)
                            rect.size.height /= screenRatio
                        }

                        rect = VNImageRectForNormalizedRect(rect, Int(screenWidth), Int(newHeight))
                        print("rect: \(rect)")
                        
                        // The labels array is a list of VNClassificationObservation objects,
                        // with the highest scoring class first in the list.
                        let label = prediction.labels[0].identifier
                        let index = self.labels.firstIndex(of: label) ?? 0
                        let confidence = prediction.labels[0].confidence
                        recognitions.append(["label": label,
                                             "confidence": confidence,
                                             "index": index,
                                             "x": rect.origin.x,
                                             "y": rect.origin.y,
                                             "width": rect.size.width,
                                             "height": rect.size.height])
                    }
                }
            }
        } catch {
            print(error)
        }
        
        completion(recognitions)
    }
}
