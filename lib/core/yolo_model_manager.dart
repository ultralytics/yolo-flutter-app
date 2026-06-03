// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';

import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/models/yolo_exceptions.dart';
import 'package:ultralytics_yolo/utils/error_handler.dart';
import 'package:ultralytics_yolo/utils/logger.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';

/// Progress snapshot for an in-flight model download.
class DownloadProgress {
  const DownloadProgress({required this.modelId, required this.fraction});

  /// Official model id (e.g. `yolo26n`, `yolo26s-seg`) being downloaded.
  final String modelId;

  /// Completion fraction in `[0, 1]`. Emits `0.0` at download start and reaches `1.0` on completion; intermediate
  /// fractions stream only when the remote `Content-Length` is known.
  final double fraction;

  @override
  String toString() =>
      'DownloadProgress(modelId: $modelId, fraction: ${fraction.toStringAsFixed(2)})';
}

class YOLOModelManager {
  // Broadcast sink for in-progress model downloads. Fed by the resolver's download path; consumers (e.g.
  // YOLOShowcase) subscribe to surface a `LinearProgressIndicator` overlay on the active model chip. Kept as a
  // singleton because downloads are global (one URL, one filesystem path) and the resolver is also stateless.
  static final StreamController<DownloadProgress> _downloadProgressController =
      StreamController<DownloadProgress>.broadcast();
  static final Set<String> _cancelledDownloads = <String>{};
  static final Map<
    String,
    ({Object token, void Function() cancel, bool cancelled})
  >
  _downloadCancelers =
      <String, ({Object token, void Function() cancel, bool cancelled})>{};

  /// Stream of in-flight model-download progress events.
  static Stream<DownloadProgress> get downloadProgress =>
      _downloadProgressController.stream;

  /// Emits a [DownloadProgress] event. Called by the model resolver during streaming downloads; safe to call from
  /// any isolate that runs the resolver (Flutter UI isolate in practice).
  static void emitProgress(String modelId, double fraction) {
    if (_downloadProgressController.isClosed) return;
    _downloadProgressController.add(
      DownloadProgress(modelId: modelId, fraction: fraction.clamp(0.0, 1.0)),
    );
  }

  static void clearDownloadCancellation(String modelId) {
    _cancelledDownloads.remove(modelId);
  }

  static Object registerDownload(String modelId, void Function() cancel) {
    final token = Object();
    final cancelled = _cancelledDownloads.remove(modelId);
    _downloadCancelers[modelId] = (
      token: token,
      cancel: cancel,
      cancelled: cancelled,
    );
    if (cancelled) cancel();
    return token;
  }

  static void cancelDownload(String modelId) {
    final download = _downloadCancelers[modelId];
    if (download == null) {
      _cancelledDownloads.add(modelId);
      return;
    }
    _downloadCancelers[modelId] = (
      token: download.token,
      cancel: download.cancel,
      cancelled: true,
    );
    download.cancel();
  }

  static bool isDownloadCancelled(String modelId, Object token) {
    final download = _downloadCancelers[modelId];
    return download?.token == token && download?.cancelled == true;
  }

  static void finishDownload(String modelId, Object token) {
    if (_downloadCancelers[modelId]?.token == token) {
      _downloadCancelers.remove(modelId);
    }
  }

  final MethodChannel _channel;
  final String _instanceId;
  final String _modelPath;
  final YOLOTask _task;
  final bool _useGpu;
  final int _numItemsThreshold;
  final Map<String, dynamic>? _classifierOptions;
  int? _viewId;

  bool _isInitialized = false;

  YOLOModelManager({
    required MethodChannel channel,
    required String instanceId,
    required String modelPath,
    required YOLOTask task,
    required bool useGpu,
    Map<String, dynamic>? classifierOptions,
    int? viewId,
    int? numItemsThreshold,
  }) : _channel = channel,
       _instanceId = instanceId,
       _modelPath = modelPath,
       _task = task,
       _useGpu = useGpu,
       _classifierOptions = classifierOptions,
       _viewId = viewId,
       _numItemsThreshold = numItemsThreshold ?? 30;

  Future<void> initializeInstance() async {
    try {
      if (_instanceId != 'default') {
        final defaultChannel = ChannelConfig.createSingleImageChannel();
        await defaultChannel.invokeMethod('createInstance', {
          'instanceId': _instanceId,
        });
      }
      _isInitialized = true;
    } catch (e) {
      throw ModelLoadingException('Failed to initialize YOLO instance: $e');
    }
  }

  Future<void> predictorInstance() async {
    if (!_isInitialized) {
      await initializeInstance();
    }
    final Map<String, dynamic> arguments = {};
    if (_instanceId != 'default') {
      arguments['instanceId'] = _instanceId;
    }
    try {
      await _channel.invokeMethod('predictorInstance', arguments);
    } catch (e) {
      throw YOLOErrorHandler.handleError(
        e,
        'Failed to predictorInstance for instance $_instanceId',
      );
    }
  }

  Future<bool> loadModel() async {
    if (!_isInitialized) {
      await initializeInstance();
    }

    try {
      final Map<String, dynamic> arguments = {
        'modelPath': _modelPath,
        'task': _task.name,
        'useGpu': _useGpu,
        'numItemsThreshold': _numItemsThreshold,
      };

      if (_classifierOptions != null) {
        arguments['classifierOptions'] = _classifierOptions;
      }

      if (_instanceId != 'default') {
        arguments['instanceId'] = _instanceId;
      }

      final result = await _channel.invokeMethod('loadModel', arguments);
      return result == true;
    } catch (e) {
      throw YOLOErrorHandler.handleError(
        e,
        'Failed to load model $_modelPath for task ${_task.name}',
      );
    }
  }

  Future<void> switchModel(String newModelPath, YOLOTask newTask) async {
    if (_viewId == null) {
      throw StateError('Cannot switch model: view not initialized');
    }

    try {
      final Map<String, dynamic> arguments = {
        'viewId': _viewId,
        'modelPath': newModelPath,
        'task': newTask.name,
        'useGpu': _useGpu,
      };

      if (_instanceId != 'default') {
        arguments['instanceId'] = _instanceId;
      }

      await _channel.invokeMethod('setModel', arguments);
    } catch (e) {
      throw YOLOErrorHandler.handleError(
        e,
        'Failed to switch to model $newModelPath for task ${newTask.name}',
      );
    }
  }

  void setViewId(int viewId) {
    _viewId = viewId;
  }

  Future<void> dispose() async {
    try {
      await _channel.invokeMethod('disposeInstance', {
        'instanceId': _instanceId,
      });
    } catch (e) {
      logInfo('Failed to dispose YOLO model instance $_instanceId: $e');
    } finally {
      _isInitialized = false;
    }
  }
}
