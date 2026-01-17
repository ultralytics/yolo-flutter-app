// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';
import 'package:ultralytics_yolo/utils/logger.dart';

/// Auto-focus mode for the camera.
enum AutoFocusMode {
  /// Camera continuously adjusts focus as the scene changes.
  continuous,

  /// Camera focuses once when triggered and then locks.
  single,
}

/// Controller for managing YOLO detection settings and camera controls.
class YOLOViewController {
  MethodChannel? _methodChannel;
  int? _viewId;
  double _confidenceThreshold = 0.5;
  double _iouThreshold = 0.45;
  int _numItemsThreshold = 30;

  double get confidenceThreshold => _confidenceThreshold;
  double get iouThreshold => _iouThreshold;
  int get numItemsThreshold => _numItemsThreshold;
  bool get isInitialized => _methodChannel != null && _viewId != null;

  YOLOViewController();

  void init(MethodChannel methodChannel, int viewId) {
    _methodChannel = methodChannel;
    _viewId = viewId;
    _applyThresholds();
  }

  Future<void> _applyThresholds() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setThresholds', {
          'confidenceThreshold': _confidenceThreshold,
          'iouThreshold': _iouThreshold,
          'numItemsThreshold': _numItemsThreshold,
        });
      } catch (e) {
        logInfo('Error applying thresholds: $e');
      }
    }
  }

  Future<void> setConfidenceThreshold(double threshold) async {
    _confidenceThreshold = threshold.clamp(0.0, 1.0);
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setConfidenceThreshold', {
          'threshold': _confidenceThreshold,
        });
      } catch (e) {
        logInfo('Error setting confidence threshold: $e');
      }
    }
  }

  Future<void> setIoUThreshold(double threshold) async {
    _iouThreshold = threshold.clamp(0.0, 1.0);
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setIoUThreshold', {
          'threshold': _iouThreshold,
        });
      } catch (e) {
        logInfo('Error setting IoU threshold: $e');
      }
    }
  }

  Future<void> setNumItemsThreshold(int numItems) async {
    _numItemsThreshold = numItems.clamp(1, 100);
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setNumItemsThreshold', {
          'numItems': _numItemsThreshold,
        });
      } catch (e) {
        logInfo('Error setting num items threshold: $e');
      }
    }
  }

  Future<void> setThresholds({
    double? confidenceThreshold,
    double? iouThreshold,
    int? numItemsThreshold,
  }) async {
    if (confidenceThreshold != null) {
      _confidenceThreshold = confidenceThreshold.clamp(0.0, 1.0);
    }
    if (iouThreshold != null) {
      _iouThreshold = iouThreshold.clamp(0.0, 1.0);
    }
    if (numItemsThreshold != null) {
      _numItemsThreshold = numItemsThreshold.clamp(1, 100);
    }

    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setThresholds', {
          'confidenceThreshold': _confidenceThreshold,
          'iouThreshold': _iouThreshold,
          'numItemsThreshold': _numItemsThreshold,
        });
      } catch (e) {
        logInfo('Error setting thresholds: $e');
      }
    }
  }

  Future<void> switchCamera() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('switchCamera');
      } catch (e) {
        logInfo('Error switching camera: $e');
      }
    }
  }

  Future<void> zoomIn() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('zoomIn');
      } catch (e) {
        logInfo('Error zooming in: $e');
      }
    }
  }

  Future<void> zoomOut() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('zoomOut');
      } catch (e) {
        logInfo('Error zooming out: $e');
      }
    }
  }

  Future<void> setZoomLevel(double zoomLevel) async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setZoomLevel', {
          'zoomLevel': zoomLevel,
        });
      } catch (e) {
        logInfo('Error setting zoom level: $e');
      }
    }
  }

  /// Toggles the torch (flashlight) on/off.
  /// Only works when the back camera is active and the device has a torch.
  Future<void> toggleTorch() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('toggleTorch');
      } catch (e) {
        logInfo('Error toggling torch: $e');
      }
    }
  }

  /// Sets the torch mode to the specified state.
  /// [enabled] - true to turn on the torch, false to turn it off.
  /// Only works when the back camera is active and the device has a torch.
  Future<void> setTorchMode(bool enabled) async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setTorchMode', {
          'enabled': enabled,
        });
      } catch (e) {
        logInfo('Error setting torch mode: $e');
      }
    }
  }

  /// Checks if the torch (flashlight) is available on the current camera.
  /// Returns true if torch is available, false otherwise.
  /// Returns null if the check fails or the view is not initialized.
  Future<bool?> isTorchAvailable() async {
    if (_methodChannel != null) {
      try {
        final result = await _methodChannel!.invokeMethod('isTorchAvailable');
        return result as bool?;
      } catch (e) {
        logInfo('Error checking torch availability: $e');
        return null;
      }
    }
    return null;
  }

  /// Gets the current torch state.
  /// Returns true if torch is currently on, false if off.
  /// Returns null if the check fails or the view is not initialized.
  Future<bool?> isTorchEnabled() async {
    if (_methodChannel != null) {
      try {
        final result = await _methodChannel!.invokeMethod('isTorchEnabled');
        return result as bool?;
      } catch (e) {
        logInfo('Error getting torch state: $e');
        return null;
      }
    }
    return null;
  }

  // ============================================================================
  // Focus Control
  // ============================================================================

  /// Sets the focus point to the specified normalized coordinates.
  /// [x] and [y] should be between 0.0 and 1.0, where (0,0) is top-left
  /// and (1,1) is bottom-right of the preview.
  /// This triggers a one-time focus at the specified point.
  Future<void> setFocusPoint(double x, double y) async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setFocusPoint', {
          'x': x.clamp(0.0, 1.0),
          'y': y.clamp(0.0, 1.0),
        });
      } catch (e) {
        logInfo('Error setting focus point: $e');
      }
    }
  }

  /// Locks the focus at the current position.
  /// Useful for batch scanning cards at a fixed distance.
  Future<void> lockFocus() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('lockFocus');
      } catch (e) {
        logInfo('Error locking focus: $e');
      }
    }
  }

  /// Unlocks the focus and returns to auto-focus mode.
  Future<void> unlockFocus() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('unlockFocus');
      } catch (e) {
        logInfo('Error unlocking focus: $e');
      }
    }
  }

  /// Sets the auto-focus mode.
  /// [mode] - Either continuous (constantly refocusing) or single (focus once).
  Future<void> setAutoFocusMode(AutoFocusMode mode) async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setAutoFocusMode', {
          'mode': mode.name,
        });
      } catch (e) {
        logInfo('Error setting auto-focus mode: $e');
      }
    }
  }

  // ============================================================================
  // Exposure Control
  // ============================================================================

  /// Sets the exposure metering point to the specified normalized coordinates.
  /// [x] and [y] should be between 0.0 and 1.0, where (0,0) is top-left
  /// and (1,1) is bottom-right of the preview.
  Future<void> setExposurePoint(double x, double y) async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setExposurePoint', {
          'x': x.clamp(0.0, 1.0),
          'y': y.clamp(0.0, 1.0),
        });
      } catch (e) {
        logInfo('Error setting exposure point: $e');
      }
    }
  }

  /// Locks the exposure at the current level.
  /// Useful for consistent exposure when scanning multiple cards.
  Future<void> lockExposure() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('lockExposure');
      } catch (e) {
        logInfo('Error locking exposure: $e');
      }
    }
  }

  /// Unlocks the exposure and returns to auto-exposure mode.
  Future<void> unlockExposure() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('unlockExposure');
      } catch (e) {
        logInfo('Error unlocking exposure: $e');
      }
    }
  }

  /// Sets the exposure compensation value in stops.
  /// [stops] - Typically ranges from -2.0 to +2.0.
  /// Positive values brighten the image, negative values darken it.
  /// Useful for adjusting exposure when scanning glossy cards.
  Future<void> setExposureCompensation(double stops) async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setExposureCompensation', {
          'stops': stops,
        });
      } catch (e) {
        logInfo('Error setting exposure compensation: $e');
      }
    }
  }

  /// Gets the supported exposure compensation range.
  /// Returns a map with 'min' and 'max' values, or null if unavailable.
  Future<Map<String, double>?> getExposureCompensationRange() async {
    if (_methodChannel != null) {
      try {
        final result =
            await _methodChannel!.invokeMethod('getExposureCompensationRange');
        if (result is Map) {
          return {
            'min': (result['min'] as num?)?.toDouble() ?? -2.0,
            'max': (result['max'] as num?)?.toDouble() ?? 2.0,
          };
        }
      } catch (e) {
        logInfo('Error getting exposure compensation range: $e');
      }
    }
    return null;
  }

  // ============================================================================
  // White Balance Control
  // ============================================================================

  /// Locks the white balance at the current setting.
  /// Useful for consistent color reproduction when scanning cards.
  Future<void> lockWhiteBalance() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('lockWhiteBalance');
      } catch (e) {
        logInfo('Error locking white balance: $e');
      }
    }
  }

  /// Unlocks the white balance and returns to auto white balance mode.
  Future<void> unlockWhiteBalance() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('unlockWhiteBalance');
      } catch (e) {
        logInfo('Error unlocking white balance: $e');
      }
    }
  }

  // ============================================================================
  // Combined Focus and Exposure (Tap-to-Focus with Exposure)
  // ============================================================================

  /// Sets both focus and exposure point to the same location.
  /// This is the typical behavior for a "tap-to-focus" interaction.
  /// [x] and [y] should be between 0.0 and 1.0.
  /// [autoReset] - If true, automatically returns to continuous AF after focusing.
  Future<void> setFocusAndExposurePoint(double x, double y,
      {bool autoReset = true}) async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setFocusAndExposurePoint', {
          'x': x.clamp(0.0, 1.0),
          'y': y.clamp(0.0, 1.0),
          'autoReset': autoReset,
        });
      } catch (e) {
        logInfo('Error setting focus and exposure point: $e');
      }
    }
  }

  /// Resets all camera controls (focus, exposure, white balance) to automatic mode.
  Future<void> resetCameraControls() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('resetCameraControls');
      } catch (e) {
        logInfo('Error resetting camera controls: $e');
      }
    }
  }

  Future<void> switchModel(String modelPath, YOLOTask task) async {
    if (_methodChannel != null && _viewId != null) {
      await _methodChannel!.invokeMethod('setModel', {
        'modelPath': modelPath,
        'task': task.name,
      });
    }
  }

  Future<void> setStreamingConfig(YOLOStreamingConfig config) async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setStreamingConfig', {
          'includeDetections': config.includeDetections,
          'includeClassifications': config.includeClassifications,
          'includeProcessingTimeMs': config.includeProcessingTimeMs,
          'includeFps': config.includeFps,
          'includeMasks': config.includeMasks,
          'includePoses': config.includePoses,
          'includeOBB': config.includeOBB,
          'includeOriginalImage': config.includeOriginalImage,
          'maxFPS': config.maxFPS,
          'throttleIntervalMs': config.throttleInterval?.inMilliseconds,
          'inferenceFrequency': config.inferenceFrequency,
          'skipFrames': config.skipFrames,
        });
      } catch (e) {
        logInfo('Error setting streaming config: $e');
      }
    }
  }

  Future<void> stop() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('stop');
      } catch (e) {
        logInfo('Error stopping: $e');
      }
    }
  }

  Future<void> restartCamera() async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('restartCamera');
      } catch (e) {
        logInfo('Error restarting camera: $e');
      }
    }
  }

  Future<void> setShowUIControls(bool show) async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setShowUIControls', {'show': show});
      } catch (e) {
        logInfo('Error setting UI controls: $e');
      }
    }
  }

  Future<void> setShowOverlays(bool show) async {
    if (_methodChannel != null) {
      try {
        await _methodChannel!.invokeMethod('setShowOverlays', {'show': show});
      } catch (e) {
        logInfo('Error setting overlay visibility: $e');
      }
    }
  }

  Future<Uint8List?> captureFrame() async {
    if (_methodChannel != null) {
      try {
        final result = await _methodChannel!.invokeMethod('captureFrame');
        return result is Uint8List ? result : null;
      } catch (e) {
        logInfo('Error capturing frame: $e');
        return null;
      }
    }
    return null;
  }
}
