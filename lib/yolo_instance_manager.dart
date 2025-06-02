// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'yolo.dart';

/// Manages multiple YOLO instances with unique IDs
class YOLOInstanceManager {
  static final Map<String, YOLO> _instances = {};

  /// Internal method to register an instance
  static void registerInstance(String instanceId, YOLO instance) {
    _instances[instanceId] = instance;
  }

  /// Internal method to unregister an instance
  static void unregisterInstance(String instanceId) {
    _instances.remove(instanceId);
  }

  /// Gets a YOLO instance by ID
  static YOLO? getInstance(String instanceId) {
    return _instances[instanceId];
  }

  /// Gets all active instance IDs
  static List<String> getActiveInstanceIds() {
    return _instances.keys.toList();
  }

  /// Checks if an instance ID exists
  static bool hasInstance(String instanceId) {
    return _instances.containsKey(instanceId);
  }
}
