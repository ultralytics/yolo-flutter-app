// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

/// Base exception class for all YOLO-related exceptions.
///
/// This is the parent class for all exceptions that can be thrown by the YOLO plugin.
/// Applications can catch this exception type to handle all YOLO-related errors in one place.
class YOLOException implements Exception {
  /// A human-readable error message
  final String message;

  /// Creates a new YOLOException with the given error message
  YOLOException(this.message);

  @override
  String toString() => 'YOLOException: $message';
}

/// Exception thrown when a model fails to load.
///
/// This exception is thrown by [YOLO.loadModel] when the model file cannot be found,
/// is in an invalid format, or is otherwise incompatible.
class ModelLoadingException extends YOLOException {
  ModelLoadingException(super.message);

  @override
  String toString() => 'ModelLoadingException: $message';
}

/// Exception thrown when attempting to perform inference without loading a model.
///
/// This exception is thrown by [YOLO.predict] when the model has not been loaded
/// or was not loaded successfully.
class ModelNotLoadedException extends YOLOException {
  ModelNotLoadedException(super.message);

  @override
  String toString() => 'ModelNotLoadedException: $message';
}

/// Exception thrown when invalid input is provided to YOLO methods.
///
/// This exception is thrown when inputs such as image data are invalid,
/// corrupted, or in an unsupported format.
class InvalidInputException extends YOLOException {
  InvalidInputException(super.message);

  @override
  String toString() => 'InvalidInputException: $message';
}

/// Exception thrown when an error occurs during model inference.
///
/// This exception is thrown by [YOLO.predict] when the model encounters an error
/// during the inference process.
class InferenceException extends YOLOException {
  InferenceException(super.message);

  @override
  String toString() => 'InferenceException: $message';
}
