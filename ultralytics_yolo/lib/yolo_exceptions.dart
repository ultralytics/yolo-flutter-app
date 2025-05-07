/// Base exception class for all YOLO-related exceptions.
///
/// This is the parent class for all exceptions that can be thrown by the YOLO plugin.
/// Applications can catch this exception type to handle all YOLO-related errors in one place.
class YoloException implements Exception {
  /// A human-readable error message
  final String message;
  
  /// Creates a new YoloException with the given error message
  YoloException(this.message);
  
  @override
  String toString() => 'YoloException: $message';
}

/// Exception thrown when a model fails to load.
///
/// This exception is thrown by [YOLO.loadModel] when the model file cannot be found,
/// is in an invalid format, or is otherwise incompatible.
class ModelLoadingException extends YoloException {
  ModelLoadingException(super.message);
  
  @override
  String toString() => 'ModelLoadingException: $message';
}

/// Exception thrown when attempting to perform inference without loading a model.
///
/// This exception is thrown by [YOLO.predict] when the model has not been loaded
/// or was not loaded successfully.
class ModelNotLoadedException extends YoloException {
  ModelNotLoadedException(super.message);
  
  @override
  String toString() => 'ModelNotLoadedException: $message';
}

/// Exception thrown when invalid input is provided to YOLO methods.
///
/// This exception is thrown when inputs such as image data are invalid,
/// corrupted, or in an unsupported format.
class InvalidInputException extends YoloException {
  InvalidInputException(super.message);
  
  @override
  String toString() => 'InvalidInputException: $message';
}

/// Exception thrown when an error occurs during model inference.
///
/// This exception is thrown by [YOLO.predict] when the model encounters an error
/// during the inference process.
class InferenceException extends YoloException {
  InferenceException(super.message);
  
  @override
  String toString() => 'InferenceException: $message';
}