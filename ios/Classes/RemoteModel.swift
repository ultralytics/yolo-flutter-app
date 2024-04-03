import CoreML

public class RemoteModel: YoloModel{
    public var task: String
    var modelUrl: String
    
    public init(modelUrl: String, task: String) {
        self.task = task
        self.modelUrl = modelUrl
    }
    
    public func loadModel() async throws -> MLModel?{
        // Define a random URL for the Core ML model definition file.
        let modelDefinitionURL = URL(string: modelUrl)!
        
        let compiledModelName = modelDefinitionURL.lastPathComponent
        let tempModelUrl = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(compiledModelName)

        // Check if the file already exists before copying it to the permanent location.
        if !FileManager.default.fileExists(atPath: tempModelUrl.path) {
            // Create a URLSession to download the file from the given URL.
            let session = URLSession.shared
            let (data, _) = try await session.data(from: modelDefinitionURL)
            
            // Save the downloaded model definition data to a temporary file URL.
            try data.write(to: tempModelUrl)
            
            // Compile the model definition file using the compileModel(at:) method.
            let compiledModelURL = try! MLModel.compileModel(at: tempModelUrl)
            
            return try! MLModel(contentsOf: compiledModelURL)
        }
        
        let compiledModelURL = try! MLModel.compileModel(at: tempModelUrl)
        return try! MLModel(contentsOf: compiledModelURL)
    }
}
