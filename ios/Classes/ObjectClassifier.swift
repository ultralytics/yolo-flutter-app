import Vision
import UIKit

public class ObjectClassifier: Predictor{
    private var classifier: VNCoreMLModel!
    private var visionRequest: VNCoreMLRequest?
    private var currentBuffer: CVPixelBuffer?
    private var currentOnResultsListener: ResultsListener?
    private var currentOnInferenceTimeListener: InferenceTimeListener?
    private var currentOnFpsRateListener: FpsRateListener?
    private var labels = [String]()
    private var t0 = 0.0  // inference start
    private var t1 = 0.0  // inference dt
    private var t2 = 0.0  // inference dt smoothed
    private var t3 = CACurrentMediaTime()  // FPS start
    private var t4 = 0.0  // FPS dt smoothed
    
    public init (yoloModel: any YoloModel) async throws{
        if(yoloModel.task != "classify") {
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
        
        
        classifier = try! VNCoreMLModel(for: mlModel)
        
        visionRequest = {
            let request = VNCoreMLRequest(model: classifier, completionHandler: {
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
            if UIDevice.current.orientation != .faceUp {  // stop if placed down on a table
                t0 = CACurrentMediaTime()  // inference start
                do {
                    try handler.perform([visionRequest!])
                } catch {
                    print(error)
                }
                t1 = CACurrentMediaTime() - t0  // inference dt
            }
            
            currentBuffer = nil
        }
    }
    
    private func processObservations(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async{
            if let observation = request.results as? [VNCoreMLFeatureValueObservation]{
                
                // Get the MLMultiArray from the observation
                let multiArray = observation.first?.featureValue.multiArrayValue

                if let multiArray = multiArray {
                    // Initialize an array to store the classes
                    var valuesArray = [Double]()

                    // Loop through the MLMultiArray and append its values to the array
                    for i in 0..<multiArray.count {
                        let value = multiArray[i].doubleValue
                        valuesArray.append(value)
                    }

                    // Create an indexed map as a dictionary
                    var indexedMap = [Int: Double]()
                    for (index, value) in valuesArray.enumerated() {
                        indexedMap[index] = value
                    }

                    // Sort the dictionary in descending order based on values
                    let sortedMap = indexedMap.sorted(by: { $0.value > $1.value })
                    
                    var recognitions: [[String:Any]] = []
                    for (index, value) in sortedMap {
                        let label = self.labels[index]
                        recognitions.append(["label": label,
                                             "confidence": value,
                                             "index": index])
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
                    
                } else {
                    print("Failed to extract MLMultiArray from the observation.")
                }
                
            }
        }
    }
    
    public func predictOnImage(image: CIImage, completion: ([[String : Any]]) -> Void) {
        let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
        let request = VNCoreMLRequest(model: classifier)
        var recognitions: [[String:Any]] = []
        
        do {
            try requestHandler.perform([request])
            if let observation = request.results as? [VNCoreMLFeatureValueObservation]{
                
                // Get the MLMultiArray from the observation
                let multiArray = observation.first?.featureValue.multiArrayValue

                if let multiArray = multiArray {
                    // Initialize an array to store the classes
                    var valuesArray = [Double]()
                    
                    // Loop through the MLMultiArray and append its values to the array
                    for i in 0..<multiArray.count {
                        let value = multiArray[i].doubleValue
                        valuesArray.append(value)
                    }

                    // Create an indexed map as a dictionary
                    var indexedMap = [Int: Double]()
                    for (index, value) in valuesArray.enumerated() {
                        indexedMap[index] = value
                    }

                    // Sort the dictionary in descending order based on values
                    let sortedMap = indexedMap.sorted(by: { $0.value > $1.value })
                    
                    for (index, value) in sortedMap {
                        let label = self.labels[index]
                        recognitions.append(["label": label,
                                             "confidence": value,
                                             "index": index])
                    }
                }
            }
        } catch{
            print(error)
        }
        
        completion(recognitions)
    }
    
}
