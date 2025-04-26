import 'package:ultralytics_yolo/predict/predictor.dart';
import 'package:ultralytics_yolo/predict/segment/detected_segment.dart';
import 'package:ultralytics_yolo/yolo_model.dart';

/// A predictor for object detection.
class SegmentDetector extends Predictor {
  /// Constructor to create an instance of [SegmentDetector].
  SegmentDetector({required YoloModel model}) : super(model);

  /// The platform instance used to run detection.
  Stream<List<DetectedSegment?>?> get detectionResultStream =>
      super.ultralyticsYoloPlatform.segmentResultStream;

  /// Sets the confidence threshold for the detection.
  void setConfidenceThreshold(double confidence) {
    super.ultralyticsYoloPlatform.setConfidenceThreshold(confidence);
  }

  /// Sets the Intersection over Union (IoU) threshold for the detection.
  void setIouThreshold(double iou) {
    super.ultralyticsYoloPlatform.setIouThreshold(iou);
  }

  /// Sets the number of items threshold for the detection.
  void setNumItemsThreshold(int numItems) {
    super.ultralyticsYoloPlatform.setNumItemsThreshold(numItems);
  }

  /// Detects objects from the given [imagePath].
  Future<List<DetectedSegment?>?> detect({required String imagePath}) =>
      super.ultralyticsYoloPlatform.segmentImage(imagePath);
}
