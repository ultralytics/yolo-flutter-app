// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/models/yolo_result.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/widgets/yolo_controller.dart';

class YOLOTestHelpers {
  static MethodChannel setupMockChannel({
    String channelName = 'yolo_single_image_channel',
    Map<String, dynamic Function(MethodCall)?>? customResponses,
  }) {
    final channel = MethodChannel(channelName);
    const pathProviderChannel = MethodChannel(
      'plugins.flutter.io/path_provider',
    );
    final log = <MethodCall>[];
    bool modelLoaded = false;

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(pathProviderChannel, (
          MethodCall methodCall,
        ) async {
          switch (methodCall.method) {
            case 'getApplicationDocumentsDirectory':
              return '/tmp/yolo_test';
            default:
              return null;
          }
        });

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (MethodCall methodCall) async {
          log.add(methodCall);

          if (customResponses != null &&
              customResponses.containsKey(methodCall.method)) {
            final customHandler = customResponses[methodCall.method];
            if (customHandler != null) {
              return customHandler(methodCall);
            }
          }

          switch (methodCall.method) {
            case 'loadModel':
              modelLoaded = true;
              return true;
            case 'createInstance':
              return true;
            case 'disposeInstance':
              return true;
            case 'predictSingleImage':
              if (!modelLoaded) {
                throw PlatformException(
                  code: 'MODEL_NOT_LOADED',
                  message: 'Model not loaded',
                );
              }
              return createMockDetectionResult();
            case 'setModel':
              return true;
            case 'setThresholds':
              return true;
            case 'setConfidenceThreshold':
              return true;
            case 'setIoUThreshold':
              return true;
            case 'setNumItemsThreshold':
              return true;
            case 'switchCamera':
              return true;
            case 'setTorchMode':
              return true;
            case 'zoomIn':
              return true;
            case 'zoomOut':
              return true;
            case 'setZoomLevel':
              return true;
            case 'captureFrame':
              return Uint8List.fromList(List.filled(100, 0));
            case 'checkModelExists':
              return {
                'exists': true,
                'path': methodCall.arguments['modelPath'],
                'location': 'assets',
              };
            case 'inspectModel':
              return {
                'path': methodCall.arguments['modelPath'],
                'task': 'detect',
                'labels': ['person'],
              };
            case 'getStoragePaths':
              return {
                'internal': '/data/data/com.example.app/files',
                'cache': '/data/data/com.example.app/cache',
                'external': null,
                'externalCache': null,
              };
            default:
              return null;
          }
        });

    return channel;
  }

  static Map<String, dynamic> createMockDetectionResult({
    int numDetections = 1,
    bool includeKeypoints = false,
    bool includeMask = false,
  }) {
    final boxes = <Map<String, dynamic>>[];
    final detections = <Map<String, dynamic>>[];

    for (int i = 0; i < numDetections; i++) {
      final box = {
        'class': 'person',
        'confidence': 0.95 - (i * 0.1),
        'x1': 10.0 + (i * 20),
        'y1': 10.0 + (i * 20),
        'x2': 110.0 + (i * 20),
        'y2': 210.0 + (i * 20),
        'x1_norm': 0.1 + (i * 0.1),
        'y1_norm': 0.1 + (i * 0.1),
        'x2_norm': 0.2 + (i * 0.1),
        'y2_norm': 0.3 + (i * 0.1),
      };
      boxes.add(box);

      final detection = {
        'classIndex': 0,
        'className': 'person',
        'confidence': 0.95 - (i * 0.1),
        'boundingBox': {
          'left': 10.0 + (i * 20),
          'top': 10.0 + (i * 20),
          'right': 110.0 + (i * 20),
          'bottom': 210.0 + (i * 20),
        },
        'normalizedBox': {
          'left': 0.1 + (i * 0.1),
          'top': 0.1 + (i * 0.1),
          'right': 0.2 + (i * 0.1),
          'bottom': 0.3 + (i * 0.1),
        },
      };

      if (includeKeypoints) {
        detection['keypoints'] = _createMockKeypoints();
      }

      if (includeMask) {
        detection['mask'] = _createMockMask();
      }

      detections.add(detection);
    }

    return {
      'boxes': boxes,
      'detections': detections,
      'annotatedImage': Uint8List.fromList(List.filled(100, 0)),
      'processingTimeMs': 50.0,
    };
  }

  static List<double> _createMockKeypoints() {
    return [
      100.0,
      150.0,
      0.9,
      90.0,
      140.0,
      0.8,
      110.0,
      140.0,
      0.8,
      80.0,
      130.0,
      0.7,
      120.0,
      130.0,
      0.7,
      70.0,
      200.0,
      0.6,
      130.0,
      200.0,
      0.6,
      60.0,
      250.0,
      0.5,
      140.0,
      250.0,
      0.5,
      50.0,
      300.0,
      0.4,
      150.0,
      300.0,
      0.4,
      80.0,
      350.0,
      0.3,
      120.0,
      350.0,
      0.3,
      70.0,
      400.0,
      0.2,
      130.0,
      400.0,
      0.2,
      60.0,
      450.0,
      0.1,
      140.0,
      450.0,
      0.1,
    ];
  }

  static List<List<double>> _createMockMask() {
    return List.generate(
      10,
      (i) => List.generate(10, (j) => (i + j) % 2 == 0 ? 1.0 : 0.0),
    );
  }

  static YOLOResult createMockYOLOResult({
    String className = 'person',
    double confidence = 0.95,
    bool includeKeypoints = false,
    bool includeMask = false,
  }) {
    final keypoints = includeKeypoints ? _createMockKeypoints() : null;
    final keypointConfidences = includeKeypoints ? List.filled(17, 0.8) : null;
    final mask = includeMask ? _createMockMask() : null;

    return YOLOResult(
      classIndex: 0,
      className: className,
      confidence: confidence,
      boundingBox: const Rect.fromLTRB(10, 10, 110, 210),
      normalizedBox: const Rect.fromLTRB(0.1, 0.1, 0.2, 0.3),
      keypoints: keypoints != null ? _keypointsToList(keypoints) : null,
      keypointConfidences: keypointConfidences,
      mask: mask,
    );
  }

  static List<Keypoint> _keypointsToList(List<double> keypoints) {
    final points = <Keypoint>[];
    for (int i = 0; i < keypoints.length; i += 3) {
      if (i + 2 < keypoints.length) {
        points.add(Keypoint(keypoints[i], keypoints[i + 1]));
      }
    }
    return points;
  }

  static Map<String, dynamic> createMockPerformanceMetrics({
    double fps = 30.0,
    double processingTimeMs = 50.0,
  }) {
    return {
      'fps': fps,
      'processingTimeMs': processingTimeMs,
      'timestamp': DateTime.now().millisecondsSinceEpoch,
    };
  }

  static Map<String, dynamic> createMockStreamingData({
    bool includeDetections = true,
    bool includePerformance = true,
    bool includeOriginalImage = false,
  }) {
    final data = <String, dynamic>{};

    if (includeDetections) {
      data['detections'] = createMockDetectionResult()['detections'];
    }

    if (includePerformance) {
      data['performance'] = createMockPerformanceMetrics();
    }

    if (includeOriginalImage) {
      data['originalImage'] = Uint8List.fromList(List.filled(100, 0));
    }

    return data;
  }

  static Map<String, dynamic> createMockTaskConfig(YOLOTask task) {
    return {
      'task': task.name,
      'modelPath': 'assets/models/yolo11n.tflite',
      'useGpu': true,
      'confidenceThreshold': 0.25,
      'iouThreshold': 0.7,
      'numItemsThreshold': 30,
    };
  }

  static Map<String, dynamic> createMockModelExistsResult({
    bool exists = true,
    String location = 'assets',
  }) {
    return {
      'exists': exists,
      'path': 'assets/models/yolo11n.tflite',
      'location': location,
    };
  }

  static Map<String, String?> createMockStoragePaths() {
    return {
      'internal': '/data/data/com.example.app/files',
      'cache': '/data/data/com.example.app/cache',
      'external': '/storage/emulated/0/Android/data/com.example.app/files',
      'externalCache': '/storage/emulated/0/Android/data/com.example.app/cache',
    };
  }

  static Uint8List storedZip(Map<String, List<int>> files) {
    final output = BytesBuilder();
    final central = BytesBuilder();
    var offset = 0;

    for (final entry in files.entries) {
      final name = utf8.encode(entry.key);
      final data = entry.value;
      final crc = _crc32(data);

      _writeU32(output, 0x04034b50);
      _writeU16(output, 20);
      _writeU16(output, 0);
      _writeU16(output, 0);
      _writeU16(output, 0);
      _writeU16(output, 0);
      _writeU32(output, crc);
      _writeU32(output, data.length);
      _writeU32(output, data.length);
      _writeU16(output, name.length);
      _writeU16(output, 0);
      output.add(name);
      output.add(data);

      _writeU32(central, 0x02014b50);
      _writeU16(central, 20);
      _writeU16(central, 20);
      _writeU16(central, 0);
      _writeU16(central, 0);
      _writeU16(central, 0);
      _writeU16(central, 0);
      _writeU32(central, crc);
      _writeU32(central, data.length);
      _writeU32(central, data.length);
      _writeU16(central, name.length);
      _writeU16(central, 0);
      _writeU16(central, 0);
      _writeU16(central, 0);
      _writeU16(central, 0);
      _writeU32(central, 0);
      _writeU32(central, offset);
      central.add(name);

      offset = output.length;
    }

    final centralBytes = central.toBytes();
    final centralOffset = output.length;
    output.add(centralBytes);
    _writeU32(output, 0x06054b50);
    _writeU16(output, 0);
    _writeU16(output, 0);
    _writeU16(output, files.length);
    _writeU16(output, files.length);
    _writeU32(output, centralBytes.length);
    _writeU32(output, centralOffset);
    _writeU16(output, 0);
    return output.toBytes();
  }

  static void assertMethodCalled(
    List<MethodCall> log,
    String method, {
    dynamic arguments,
  }) {
    final calls = log.where((call) => call.method == method).toList();
    expect(calls.isNotEmpty, true, reason: 'Method $method was not called');

    if (arguments != null) {
      final call = calls.first;
      expect(call.arguments, arguments);
    }
  }

  static void assertMethodCallCount(
    List<MethodCall> log,
    String method,
    int expectedCount,
  ) {
    final calls = log.where((call) => call.method == method).toList();
    expect(
      calls.length,
      expectedCount,
      reason: 'Expected $expectedCount calls to $method, got ${calls.length}',
    );
  }

  static void clearLog(List<MethodCall> log) {
    log.clear();
  }

  static PlatformException createMockPlatformException({
    String code = 'TEST_ERROR',
    String message = 'Test error message',
  }) {
    return PlatformException(code: code, message: message);
  }

  static Future<void> waitForCondition(
    bool Function() condition, {
    Duration timeout = const Duration(seconds: 5),
    Duration interval = const Duration(milliseconds: 100),
  }) async {
    final stopwatch = Stopwatch()..start();

    while (!condition() && stopwatch.elapsed < timeout) {
      await Future.delayed(interval);
    }

    expect(condition(), true, reason: 'Condition not met within timeout');
  }

  static (MethodChannel, List<MethodCall>) createYOLOTestSetup({
    String? channelName,
    Map<String, dynamic Function(MethodCall)?>? customResponses,
  }) {
    final log = <MethodCall>[];
    final channel = setupMockChannel(
      channelName: channelName ?? 'yolo_single_image_channel',
      customResponses:
          customResponses ??
          {
            'loadModel': (call) {
              log.add(call);
              return true;
            },
            'createInstance': (call) {
              log.add(call);
              return true;
            },
            'disposeInstance': (call) {
              log.add(call);
              return true;
            },
            'predictSingleImage': (call) {
              log.add(call);
              return createMockDetectionResult();
            },
            'setModel': (call) {
              log.add(call);
              return true;
            },
            'setThresholds': (call) {
              log.add(call);
              return true;
            },
            'setConfidenceThreshold': (call) {
              log.add(call);
              return true;
            },
            'setIoUThreshold': (call) {
              log.add(call);
              return true;
            },
            'setNumItemsThreshold': (call) {
              log.add(call);
              return true;
            },
            'switchCamera': (call) {
              log.add(call);
              return true;
            },
            'setTorchMode': (call) {
              log.add(call);
              // Echo the requested state, mirroring the native handlers which return the actual resulting state.
              return (call.arguments as Map)['enabled'] as bool;
            },
            'zoomIn': (call) {
              log.add(call);
              return true;
            },
            'zoomOut': (call) {
              log.add(call);
              return true;
            },
            'setZoomLevel': (call) {
              log.add(call);
              return true;
            },
            'setShowOverlays': (call) {
              log.add(call);
              return true;
            },
            'captureFrame': (call) {
              log.add(call);
              return Uint8List.fromList(List.filled(100, 0));
            },
            'checkModelExists': (call) {
              log.add(call);
              return createMockModelExistsResult();
            },
            'inspectModel': (call) {
              log.add(call);
              return {
                'path': call.arguments['modelPath'],
                'task': 'detect',
                'labels': ['person'],
              };
            },
            'getStoragePaths': (call) {
              log.add(call);
              return createMockStoragePaths();
            },
          },
    );
    return (channel, log);
  }

  static void validateThresholdBehavior(
    YOLOViewController controller,
    List<MethodCall> log,
    MethodChannel channel,
  ) {
    controller.setConfidenceThreshold(0.8);
    expect(controller.confidenceThreshold, 0.8);
    assertMethodCalled(
      log,
      'setConfidenceThreshold',
      arguments: {'threshold': 0.8},
    );

    controller.setIoUThreshold(0.6);
    expect(controller.iouThreshold, 0.6);
    assertMethodCalled(log, 'setIoUThreshold', arguments: {'threshold': 0.6});

    controller.setNumItemsThreshold(50);
    expect(controller.numItemsThreshold, 50);
    assertMethodCalled(
      log,
      'setNumItemsThreshold',
      arguments: {'numItems': 50},
    );
  }

  static void _writeU16(BytesBuilder builder, int value) {
    builder.add(<int>[value & 0xff, (value >> 8) & 0xff]);
  }

  static void _writeU32(BytesBuilder builder, int value) {
    builder.add(<int>[
      value & 0xff,
      (value >> 8) & 0xff,
      (value >> 16) & 0xff,
      (value >> 24) & 0xff,
    ]);
  }

  static int _crc32(List<int> data) {
    var crc = 0xffffffff;
    for (final byte in data) {
      var value = (crc ^ byte) & 0xff;
      for (var bit = 0; bit < 8; bit++) {
        value = (value & 1) != 0 ? (0xedb88320 ^ (value >> 1)) : (value >> 1);
      }
      crc = value ^ (crc >> 8);
    }
    return (crc ^ 0xffffffff) & 0xffffffff;
  }
}
