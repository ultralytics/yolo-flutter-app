import 'package:plugin_platform_interface/plugin_platform_interface.dart';
import 'package:ultralytics_yolo/predict/classify/classification_result.dart';
import 'package:ultralytics_yolo/predict/detect/detected_object.dart';
import 'package:ultralytics_yolo/ultralytics_yolo_platform_channel.dart';

/// The interface that implementations of ultralytics_yolo must implement.
abstract class UltralyticsYoloPlatform extends PlatformInterface {
  /// Constructs a UltralyticsYoloPlatform.
  UltralyticsYoloPlatform() : super(token: _token);

  static final Object _token = Object();

  static UltralyticsYoloPlatform _instance = PlatformChannelUltralyticsYolo();

  /// The default instance of [UltralyticsYoloPlatform] to use.
  ///
  /// Defaults to [PlatformChannelUltralyticsYolo].
  static UltralyticsYoloPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [UltralyticsYoloPlatform] when
  /// they register themselves.
  static set instance(UltralyticsYoloPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  /// Load the model from the given [model] and [useGpu].
  Future<String?> loadModel(Map<String, dynamic> model, {bool useGpu = true}) {
    throw UnimplementedError('loadModel() has not been implemented.');
  }

  /// Set the confidence threshold for the model.
  Future<String?> setConfidenceThreshold(double confidence) {
    throw UnimplementedError(
      'setConfidenceThreshold has not been implemented.',
    );
  }

  /// Set the Intersection over Union (IoU) threshold for the model.
  Future<String?> setIouThreshold(double iou) {
    throw UnimplementedError('setIoUThreshold has not been implemented.');
  }

  /// Set the number of items threshold for the model.
  Future<String?> setNumItemsThreshold(int numItems) {
    throw UnimplementedError('setNumItemsThreshold has not been implemented.');
  }

  /// Set the zoom ratio for the camera preview.
  Future<String?> setZoomRatio(double ratio) {
    throw UnimplementedError('setZoomRatio has not been implemented.');
  }

  /// Set the lens direction for the camera preview.
  Future<String?> setLensDirection(int direction) {
    throw UnimplementedError('setLensDirection has not been implemented.');
  }

  /// Close the camera.
  Future<String?> closeCamera() {
    throw UnimplementedError('closeCamera has not been implemented.');
  }

  /// Start the camera.
  Future<String?> startCamera() {
    throw UnimplementedError('startCamera has not been implemented.');
  }

  /// Start the live prediction.
  Future<String?> pauseLivePrediction() {
    throw UnimplementedError('pauseLivePrediction has not been implemented.');
  }

  /// Resume the live prediction.
  Future<String?> resumeLivePrediction() {
    throw UnimplementedError('resumeLivePrediction has not been implemented.');
  }

  /// Stream of detected objects.
  Stream<List<DetectedObject?>?> get detectionResultStream {
    throw UnimplementedError('detectionResultStream has not been implemented.');
  }

  /// Detect objects in the given [imagePath].
  Future<List<DetectedObject?>?> detectImage(String imagePath) {
    throw UnimplementedError('detectImage has not been implemented.');
  }

  /// Stream of classification results.
  Stream<List<ClassificationResult?>?> get classificationResultStream {
    throw UnimplementedError(
      'classificationResultStream has not been implemented.',
    );
  }

  /// Classify the given [imagePath].
  Future<List<ClassificationResult?>?> classifyImage(String imagePath) {
    throw UnimplementedError('predictImage has not been implemented.');
  }

  /// Stream of inference time.
  Stream<double>? get inferenceTimeStream {
    throw UnimplementedError('inferenceTimeStream has not been implemented.');
  }

  /// Stream of frames per second (FPS) rate.
  Stream<double>? get fpsRateStream {
    throw UnimplementedError('fpsRateStream has not been implemented.');
  }
}
