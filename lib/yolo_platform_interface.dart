// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'yolo_method_channel.dart';

/// The interface that implementations of the Ultralytics YOLO plugin must implement.
///
/// This class uses the [PlatformInterface] pattern to ensure that platform-specific
/// implementations properly extend this class rather than implementing it.
///
/// Platform implementations should extend this class rather than implement it as `yolo`
/// does not consider newly added methods to be breaking changes. Extending this class
/// (using `extends`) ensures that the subclass will get the default implementation, while
/// platform implementations that `implements` this interface will be broken by newly added
/// [YOLOPlatform] methods.
///
/// The plugin uses method channels for communication between Flutter and native code.
/// Each platform (iOS, Android) provides its own implementation of the YOLO inference engine.
abstract class YOLOPlatform extends PlatformInterface {
  /// Constructs a YOLOPlatform.
  YOLOPlatform() : super(token: _token);

  static final Object _token = Object();

  static YOLOPlatform _instance = YOLOMethodChannel();

  /// The default instance of [YOLOPlatform] to use.
  ///
  /// Defaults to [YOLOMethodChannel].
  static YOLOPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [YOLOPlatform] when
  /// they register themselves.
  static set instance(YOLOPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  /// Returns the current platform version.
  ///
  /// This method is primarily used for testing and debugging to verify that
  /// the method channel communication is working correctly between Flutter
  /// and the native platform.
  ///
  /// Each platform implementation should override this method to return
  /// meaningful platform information.
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
