// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_streaming_config.dart';

void main() {
  group('YOLOStreamingConfig', () {
    test('default constructor sets correct defaults', () {
      const config = YOLOStreamingConfig();

      expect(config.includeDetections, isTrue);
      expect(config.includeClassifications, isTrue);
      expect(config.includeProcessingTimeMs, isTrue);
      expect(config.includeFps, isTrue);
      expect(config.includeMasks, isFalse);
      expect(config.includePoses, isFalse);
      expect(config.includeOBB, isFalse);
      expect(config.includeOriginalImage, isFalse);
      expect(config.maxFPS, isNull);
      expect(config.throttleInterval, isNull);
      expect(config.inferenceFrequency, isNull);
      expect(config.skipFrames, isNull);
    });

    test('constructor with custom parameters', () {
      const config = YOLOStreamingConfig(
        includeDetections: false,
        includeClassifications: false,
        includeProcessingTimeMs: false,
        includeFps: false,
        includeMasks: true,
        includePoses: true,
        includeOBB: true,
        includeOriginalImage: true,
        maxFPS: 30,
        throttleInterval: Duration(milliseconds: 100),
        inferenceFrequency: 15,
        skipFrames: 2,
      );

      expect(config.includeDetections, isFalse);
      expect(config.includeClassifications, isFalse);
      expect(config.includeProcessingTimeMs, isFalse);
      expect(config.includeFps, isFalse);
      expect(config.includeMasks, isTrue);
      expect(config.includePoses, isTrue);
      expect(config.includeOBB, isTrue);
      expect(config.includeOriginalImage, isTrue);
      expect(config.maxFPS, equals(30));
      expect(
        config.throttleInterval,
        equals(const Duration(milliseconds: 100)),
      );
      expect(config.inferenceFrequency, equals(15));
      expect(config.skipFrames, equals(2));
    });

    test('minimal() factory constructor', () {
      const config = YOLOStreamingConfig.minimal();

      expect(config.includeDetections, isTrue);
      expect(config.includeClassifications, isTrue);
      expect(config.includeProcessingTimeMs, isTrue);
      expect(config.includeFps, isTrue);
      expect(config.includeMasks, isFalse);
      expect(config.includePoses, isFalse);
      expect(config.includeOBB, isFalse);
      expect(config.includeOriginalImage, isFalse);
      expect(config.maxFPS, isNull);
      expect(config.throttleInterval, isNull);
      expect(config.inferenceFrequency, isNull);
      expect(config.skipFrames, isNull);
    });

    test('custom() factory constructor with defaults', () {
      const config = YOLOStreamingConfig.custom();

      expect(config.includeDetections, isTrue);
      expect(config.includeClassifications, isTrue);
      expect(config.includeProcessingTimeMs, isTrue);
      expect(config.includeFps, isTrue);
      expect(config.includeMasks, isFalse);
      expect(config.includePoses, isFalse);
      expect(config.includeOBB, isFalse);
      expect(config.includeOriginalImage, isFalse);
    });

    test('custom() factory constructor with specified parameters', () {
      const config = YOLOStreamingConfig.custom(
        includeDetections: false,
        includeMasks: true,
        includePoses: true,
        maxFPS: 20,
        inferenceFrequency: 10,
      );

      expect(config.includeDetections, isFalse);
      expect(config.includeClassifications, isTrue); // default
      expect(config.includeMasks, isTrue);
      expect(config.includePoses, isTrue);
      expect(config.includeOBB, isFalse); // default
      expect(config.maxFPS, equals(20));
      expect(config.inferenceFrequency, equals(10));
    });

    test('withMasks() factory constructor', () {
      const config = YOLOStreamingConfig.withMasks();

      expect(config.includeDetections, isTrue);
      expect(config.includeClassifications, isTrue);
      expect(config.includeProcessingTimeMs, isTrue);
      expect(config.includeFps, isTrue);
      expect(config.includeMasks, isTrue);
      expect(config.includePoses, isFalse);
      expect(config.includeOBB, isFalse);
      expect(config.includeOriginalImage, isFalse);
    });

    test('withPoses() factory constructor', () {
      const config = YOLOStreamingConfig.withPoses();

      expect(config.includeDetections, isTrue);
      expect(config.includeClassifications, isTrue);
      expect(config.includeProcessingTimeMs, isTrue);
      expect(config.includeFps, isTrue);
      expect(config.includeMasks, isFalse);
      expect(config.includePoses, isTrue);
      expect(config.includeOBB, isFalse);
      expect(config.includeOriginalImage, isFalse);
    });

    test('full() factory constructor', () {
      const config = YOLOStreamingConfig.full();

      expect(config.includeDetections, isTrue);
      expect(config.includeClassifications, isTrue);
      expect(config.includeProcessingTimeMs, isTrue);
      expect(config.includeFps, isTrue);
      expect(config.includeMasks, isTrue);
      expect(config.includePoses, isTrue);
      expect(config.includeOBB, isTrue);
      expect(
        config.includeOriginalImage,
        isFalse,
      ); // Still false for performance
    });

    test('debug() factory constructor', () {
      const config = YOLOStreamingConfig.debug();

      expect(config.includeDetections, isTrue);
      expect(config.includeClassifications, isTrue);
      expect(config.includeProcessingTimeMs, isTrue);
      expect(config.includeFps, isTrue);
      expect(config.includeMasks, isTrue);
      expect(config.includePoses, isTrue);
      expect(config.includeOBB, isTrue);
      expect(config.includeOriginalImage, isTrue); // True for debug mode
    });

    test('throttled() factory constructor with defaults', () {
      final config = YOLOStreamingConfig.throttled(maxFPS: 15);

      expect(config.includeDetections, isTrue);
      expect(config.includeClassifications, isTrue);
      expect(config.includeProcessingTimeMs, isTrue);
      expect(config.includeFps, isTrue);
      expect(config.includeMasks, isFalse);
      expect(config.includePoses, isFalse);
      expect(config.includeOBB, isFalse);
      expect(config.includeOriginalImage, isFalse);
      expect(config.maxFPS, equals(15));
    });

    test('throttled() factory constructor with custom parameters', () {
      final config = YOLOStreamingConfig.throttled(
        maxFPS: 10,
        includeMasks: true,
        includePoses: true,
        inferenceFrequency: 5,
        skipFrames: 3,
      );

      expect(config.maxFPS, equals(10));
      expect(config.includeMasks, isTrue);
      expect(config.includePoses, isTrue);
      expect(config.inferenceFrequency, equals(5));
      expect(config.skipFrames, equals(3));
    });

    test('powerSaving() factory constructor with defaults', () {
      final config = YOLOStreamingConfig.powerSaving();

      expect(config.inferenceFrequency, equals(10));
      expect(config.maxFPS, equals(15));
      expect(config.includeDetections, isTrue);
      expect(config.includeClassifications, isTrue);
      expect(config.includeProcessingTimeMs, isTrue);
      expect(config.includeFps, isTrue);
      expect(config.includeMasks, isFalse);
      expect(config.includePoses, isFalse);
      expect(config.includeOBB, isFalse);
      expect(config.includeOriginalImage, isFalse);
    });

    test('powerSaving() factory constructor with custom parameters', () {
      final config = YOLOStreamingConfig.powerSaving(
        inferenceFrequency: 5,
        maxFPS: 10,
      );

      expect(config.inferenceFrequency, equals(5));
      expect(config.maxFPS, equals(10));
    });

    test('highPerformance() factory constructor with defaults', () {
      final config = YOLOStreamingConfig.highPerformance();

      expect(config.inferenceFrequency, equals(30));
      expect(config.includeDetections, isTrue);
      expect(config.includeClassifications, isTrue);
      expect(config.includeProcessingTimeMs, isTrue);
      expect(config.includeFps, isTrue);
      expect(config.includeMasks, isFalse);
      expect(config.includePoses, isFalse);
      expect(config.includeOBB, isFalse);
      expect(config.includeOriginalImage, isFalse);
    });

    test('highPerformance() factory constructor with custom frequency', () {
      final config = YOLOStreamingConfig.highPerformance(
        inferenceFrequency: 60,
      );

      expect(config.inferenceFrequency, equals(60));
    });

    test('toString() method', () {
      const config = YOLOStreamingConfig(
        includeDetections: true,
        includeMasks: false,
        maxFPS: 25,
        throttleInterval: Duration(milliseconds: 50),
        inferenceFrequency: 20,
        skipFrames: 1,
      );

      final stringRep = config.toString();

      expect(stringRep, contains('YOLOStreamingConfig'));
      expect(stringRep, contains('detections: true'));
      expect(stringRep, contains('masks: false'));
      expect(stringRep, contains('maxFPS: 25'));
      expect(stringRep, contains('throttleInterval: 50ms'));
      expect(stringRep, contains('inferenceFrequency: 20'));
      expect(stringRep, contains('skipFrames: 1'));
    });

    test('toString() method with null values', () {
      const config = YOLOStreamingConfig();

      final stringRep = config.toString();

      expect(stringRep, contains('maxFPS: null'));
      expect(stringRep, contains('throttleInterval: nullms'));
      expect(stringRep, contains('inferenceFrequency: null'));
      expect(stringRep, contains('skipFrames: null'));
    });

    test('different configurations are distinguishable', () {
      const config1 = YOLOStreamingConfig.minimal();
      const config2 = YOLOStreamingConfig.withMasks();
      const config3 = YOLOStreamingConfig.debug();

      expect(config1.includeMasks, isFalse);
      expect(config2.includeMasks, isTrue);
      expect(config3.includeOriginalImage, isTrue);

      expect(config1.toString(), isNot(equals(config2.toString())));
      expect(config2.toString(), isNot(equals(config3.toString())));
    });

    test('all boolean flags can be independently controlled', () {
      const config = YOLOStreamingConfig(
        includeDetections: false,
        includeClassifications: false,
        includeProcessingTimeMs: false,
        includeFps: false,
        includeMasks: false,
        includePoses: false,
        includeOBB: false,
        includeOriginalImage: false,
      );

      expect(config.includeDetections, isFalse);
      expect(config.includeClassifications, isFalse);
      expect(config.includeProcessingTimeMs, isFalse);
      expect(config.includeFps, isFalse);
      expect(config.includeMasks, isFalse);
      expect(config.includePoses, isFalse);
      expect(config.includeOBB, isFalse);
      expect(config.includeOriginalImage, isFalse);
    });

    test('custom configuration with specific settings', () {
      const config = YOLOStreamingConfig.custom(
        includeDetections: false,
        includeClassifications: true,
        includeProcessingTimeMs: false,
        includeFps: true,
        includeMasks: false,
        includePoses: true,
        includeOBB: false,
        includeOriginalImage: true,
        maxFPS: 15,
      );

      expect(config.includeDetections, false);
      expect(config.includeClassifications, true);
      expect(config.includeProcessingTimeMs, false);
      expect(config.includeFps, true);
      expect(config.includeMasks, false);
      expect(config.includePoses, true);
      expect(config.includeOBB, false);
      expect(config.includeOriginalImage, true);
      expect(config.maxFPS, 15);
    });

    test('throttled config has correct settings', () {
      final config = YOLOStreamingConfig.throttled(maxFPS: 10);

      expect(config.includeDetections, true);
      expect(config.includeClassifications, true);
      expect(config.includeProcessingTimeMs, true);
      expect(config.includeFps, true);
      expect(config.includeMasks, false);
      expect(config.includePoses, false);
      expect(config.includeOBB, false);
      expect(config.includeOriginalImage, false);
      expect(config.maxFPS, 10);
      expect(config.throttleInterval, null);
      expect(config.inferenceFrequency, null);
      expect(config.skipFrames, null);
    });

    test('custom configuration with inference control', () {
      const config = YOLOStreamingConfig.custom(
        inferenceFrequency: 3,
        skipFrames: 2,
      );

      expect(config.inferenceFrequency, 3);
      expect(config.skipFrames, 2);
      expect(config.maxFPS, null);
      expect(config.throttleInterval, isNull);
    });
  });
}
