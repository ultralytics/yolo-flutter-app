// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:plugin_platform_interface/plugin_platform_interface.dart';

/// The plugin uses method channels for communication between Flutter and native code.
/// Each platform (iOS, Android) provides its own implementation of the YOLO inference engine.
abstract class YOLOPlatform extends PlatformInterface {
  /// Constructs a YOLOPlatform.
  YOLOPlatform() : super(token: _token);

  static final Object _token = Object();

  static YOLOPlatform? _instance;

  /// The default instance of [YOLOPlatform] to use.
  ///
  /// Defaults to [YOLOMethodChannel].
  static YOLOPlatform get instance => _instance ??= _createDefaultInstance();

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [YOLOPlatform] when
  /// they register themselves.
  static set instance(YOLOPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  /// Creates the default implementation instance.
  static YOLOPlatform _createDefaultInstance() {
    throw UnimplementedError('Default implementation not available');
  }

  /// Returns the current platform version.
  ///
  /// This method is primarily used for testing and debugging to verify that
  /// the method channel communication is working correctly between Flutter
  /// and the native platform.
  ///
  /// Returns a string containing the platform name and version
  /// (e.g., "Android 12" or "iOS 15.0"), or null if unavailable.
  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }

  /// Sets the model for an existing YOLO view.
  ///
  /// This method allows switching the model on an existing YOLO view instance
  /// without recreating the entire view.
  ///
  /// Parameters:
  /// - [viewId]: The unique identifier of the YOLO view
  /// - [modelPath]: The path to the new model file
  /// - [task]: The YOLO task type for the new model
  ///
  /// Throws:
  /// - [UnimplementedError] if not implemented by the platform
  /// - Platform-specific exceptions if the model switch fails
  Future<void> setModel(int viewId, String modelPath, String task) {
    throw UnimplementedError('setModel() has not been implemented.');
  }
}
