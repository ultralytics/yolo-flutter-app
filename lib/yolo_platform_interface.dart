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
/// [YoloPlatform] methods.
///
/// The plugin uses method channels for communication between Flutter and native code.
/// Each platform (iOS, Android) provides its own implementation of the YOLO inference engine.
abstract class YoloPlatform extends PlatformInterface {
  /// Constructs a YoloPlatform.
  YoloPlatform() : super(token: _token);

  static final Object _token = Object();

  static YoloPlatform _instance = MethodChannelYolo();

  /// The default instance of [YoloPlatform] to use.
  ///
  /// Defaults to [MethodChannelYolo].
  static YoloPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [YoloPlatform] when
  /// they register themselves.
  static set instance(YoloPlatform instance) {
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
}
