// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/services.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/models/yolo_exceptions.dart';
import 'package:ultralytics_yolo/utils/error_handler.dart';
import 'package:ultralytics_yolo/utils/logger.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';

class YOLOModelManager {
  final MethodChannel _channel;
  final String _instanceId;
  final String _modelPath;
  final YOLOTask _task;
  final bool _useGpu;
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
  }) : _channel = channel,
       _instanceId = instanceId,
       _modelPath = modelPath,
       _task = task,
       _useGpu = useGpu,
       _classifierOptions = classifierOptions,
       _viewId = viewId;

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

  Future<bool> loadModel() async {
    if (!_isInitialized) {
      await initializeInstance();
    }

    try {
      final Map<String, dynamic> arguments = {
        'modelPath': _modelPath,
        'task': _task.name,
        'useGpu': _useGpu,
      };

      if (_classifierOptions != null) {
        arguments['classifierOptions'] = _classifierOptions;
      }

      if (_instanceId != 'default') {
        arguments['instanceId'] = _instanceId;
      }

      final result = await _channel.invokeMethod('loadModel', arguments);
      return result == true;
    } on PlatformException catch (e) {
      throw YOLOErrorHandler.handleError(
        e,
        'Failed to load model $_modelPath for task ${_task.name}',
      );
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
    } on PlatformException catch (e) {
      throw YOLOErrorHandler.handleError(
        e,
        'Failed to switch to model $newModelPath for task ${newTask.name}',
      );
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
