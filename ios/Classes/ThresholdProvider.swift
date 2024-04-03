//  Copyright © 2018-2021 Ultralytics LLC. All rights reserved.
//  Class providing custom thresholds for an object detection model.
import CoreML

/// - Tag: ThresholdProvider
/// Class providing customized thresholds for object detection model
public class ThresholdProvider: MLFeatureProvider {
    /// The actual NMS values to provide as input
    var values = [
        "iouThreshold": MLFeatureValue(double: 0.0),
        "confidenceThreshold": MLFeatureValue(double: 0.0)
    ]

    /// The feature names the provider has, per the MLFeatureProvider protocol
    public var featureNames: Set<String> {
        return Set(values.keys)
    }

    /// Initialize with default values
    init(iouThreshold: Double = 0.4, confidenceThreshold: Double = 0.2) {
        self.values["iouThreshold"] = MLFeatureValue(double: iouThreshold)
        self.values["confidenceThreshold"] = MLFeatureValue(double: confidenceThreshold)
    }

    /// The actual values for the features the provider can provide
    public func featureValue(for featureName: String) -> MLFeatureValue? {
        return values[featureName]
    }
}
