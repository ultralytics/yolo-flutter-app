import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'yolo_method_channel.dart';

/// The interface that implementations of yolo must implement.
///
/// Platform implementations should extend this class rather than implement it as `yolo`
/// does not consider newly added methods to be breaking changes. Extending this class
/// (using `extends`) ensures that the subclass will get the default implementation, while
/// platform implementations that `implements` this interface will be broken by newly added
/// [YoloPlatform] methods.
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
  /// This method is used to test that method channels are working correctly.
  /// Each platform implementation should override this method to return the platform version.
  ///
  /// @return The platform version as a string (e.g., "Android 12" or "iOS 15.0").
  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
