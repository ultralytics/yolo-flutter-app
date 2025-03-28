// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import CoreML

public protocol YoloModel {
  associatedtype MLModel

  var task: String { get set }

  func loadModel() async throws -> MLModel?
}
