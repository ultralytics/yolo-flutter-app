import CoreML

public protocol YoloModel{
    associatedtype MLModel
    
    var task: String { get set }

    func loadModel() async throws -> MLModel?
}
