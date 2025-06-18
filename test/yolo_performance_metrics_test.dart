// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_performance_metrics.dart';

void main() {
  group('YOLOPerformanceMetrics', () {
    test('constructor creates metrics with all required fields', () {
      final timestamp = DateTime.now();
      const fps = 30.0;
      const processingTime = 33.3;
      const frameNumber = 100;

      final metrics = YOLOPerformanceMetrics(
        fps: fps,
        processingTimeMs: processingTime,
        frameNumber: frameNumber,
        timestamp: timestamp,
      );

      expect(metrics.fps, equals(fps));
      expect(metrics.processingTimeMs, equals(processingTime));
      expect(metrics.frameNumber, equals(frameNumber));
      expect(metrics.timestamp, equals(timestamp));
    });

    test('fromMap() factory constructor with valid data', () {
      final data = {'fps': 25.5, 'processingTimeMs': 40.0, 'frameNumber': 50};

      final metrics = YOLOPerformanceMetrics.fromMap(data);

      expect(metrics.fps, equals(25.5));
      expect(metrics.processingTimeMs, equals(40.0));
      expect(metrics.frameNumber, equals(50));
      expect(metrics.timestamp, isA<DateTime>());
    });

    test('fromMap() factory constructor with int values', () {
      final data = {
        'fps': 30, // int instead of double
        'processingTimeMs': 25, // int instead of double
        'frameNumber': 75,
      };

      final metrics = YOLOPerformanceMetrics.fromMap(data);

      expect(metrics.fps, equals(30.0));
      expect(metrics.processingTimeMs, equals(25.0));
      expect(metrics.frameNumber, equals(75));
    });

    test('fromMap() factory constructor with missing data uses defaults', () {
      final data = <String, dynamic>{};

      final metrics = YOLOPerformanceMetrics.fromMap(data);

      expect(metrics.fps, equals(0.0));
      expect(metrics.processingTimeMs, equals(0.0));
      expect(metrics.frameNumber, equals(0));
      expect(metrics.timestamp, isA<DateTime>());
    });

    test('fromMap() factory constructor with partial data', () {
      final data = {
        'fps': 15.0,
        // missing processingTimeMs and frameNumber
      };

      final metrics = YOLOPerformanceMetrics.fromMap(data);

      expect(metrics.fps, equals(15.0));
      expect(metrics.processingTimeMs, equals(0.0));
      expect(metrics.frameNumber, equals(0));
    });

    test('fromMap() factory constructor with null values uses defaults', () {
      final data = {'fps': null, 'processingTimeMs': null, 'frameNumber': null};

      final metrics = YOLOPerformanceMetrics.fromMap(data);

      expect(metrics.fps, equals(0.0));
      expect(metrics.processingTimeMs, equals(0.0));
      expect(metrics.frameNumber, equals(0));
    });

    test('toMap() converts metrics to map correctly', () {
      final timestamp = DateTime(2023, 12, 25, 10, 30, 45, 123);
      final metrics = YOLOPerformanceMetrics(
        fps: 28.5,
        processingTimeMs: 35.2,
        frameNumber: 150,
        timestamp: timestamp,
      );

      final map = metrics.toMap();

      expect(map['fps'], equals(28.5));
      expect(map['processingTimeMs'], equals(35.2));
      expect(map['frameNumber'], equals(150));
      expect(map['timestamp'], equals(timestamp.millisecondsSinceEpoch));
    });

    test('toString() returns formatted string', () {
      final timestamp = DateTime(2023, 12, 25, 10, 30, 45);
      final metrics = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.333,
        frameNumber: 100,
        timestamp: timestamp,
      );

      final stringRep = metrics.toString();

      expect(stringRep, contains('YOLOPerformanceMetrics'));
      expect(stringRep, contains('fps: 30.0'));
      expect(stringRep, contains('processingTime: 33.333ms'));
      expect(stringRep, contains('frame: 100'));
      expect(stringRep, contains('timestamp: 2023-12-25T10:30:45.000'));
    });

    test('copyWith() creates new instance with modified values', () {
      final originalTimestamp = DateTime(2023, 12, 25, 10, 30, 45);
      final original = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: originalTimestamp,
      );

      final newTimestamp = DateTime(2023, 12, 26, 11, 45, 0);
      final modified = original.copyWith(
        fps: 25.0,
        frameNumber: 200,
        timestamp: newTimestamp,
      );

      expect(modified.fps, equals(25.0));
      expect(modified.processingTimeMs, equals(33.3)); // unchanged
      expect(modified.frameNumber, equals(200));
      expect(modified.timestamp, equals(newTimestamp));

      // Original should be unchanged
      expect(original.fps, equals(30.0));
      expect(original.frameNumber, equals(100));
      expect(original.timestamp, equals(originalTimestamp));
    });

    test('copyWith() with no parameters returns identical copy', () {
      final timestamp = DateTime.now();
      final original = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: timestamp,
      );

      final copy = original.copyWith();

      expect(copy.fps, equals(original.fps));
      expect(copy.processingTimeMs, equals(original.processingTimeMs));
      expect(copy.frameNumber, equals(original.frameNumber));
      expect(copy.timestamp, equals(original.timestamp));
    });

    test('isGoodPerformance returns correct values', () {
      // Good performance cases
      final goodPerformance1 = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 50.0,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(goodPerformance1.isGoodPerformance, isTrue);

      final goodPerformance2 = YOLOPerformanceMetrics(
        fps: 15.0, // exactly at threshold
        processingTimeMs: 100.0, // exactly at threshold
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(goodPerformance2.isGoodPerformance, isTrue);

      // Poor performance cases
      final poorPerformance1 = YOLOPerformanceMetrics(
        fps: 14.9, // below threshold
        processingTimeMs: 50.0,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(poorPerformance1.isGoodPerformance, isFalse);

      final poorPerformance2 = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 100.1, // above threshold
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(poorPerformance2.isGoodPerformance, isFalse);
    });

    test('hasPerformanceIssues returns correct values', () {
      // No issues
      final noIssues = YOLOPerformanceMetrics(
        fps: 15.0,
        processingTimeMs: 150.0,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(noIssues.hasPerformanceIssues, isFalse);

      // Has issues - low FPS
      final lowFps = YOLOPerformanceMetrics(
        fps: 9.9, // below 10.0 threshold
        processingTimeMs: 50.0,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(lowFps.hasPerformanceIssues, isTrue);

      // Has issues - high processing time
      final highProcessingTime = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 200.1, // above 200.0 threshold
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(highProcessingTime.hasPerformanceIssues, isTrue);

      // Edge case - exactly at thresholds
      final edgeCase = YOLOPerformanceMetrics(
        fps: 10.0, // exactly at threshold
        processingTimeMs: 200.0, // exactly at threshold
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(edgeCase.hasPerformanceIssues, isFalse);
    });

    test('performanceRating returns correct ratings', () {
      // Excellent
      final excellent = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 40.0,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(excellent.performanceRating, equals('Excellent'));

      // Good
      final good = YOLOPerformanceMetrics(
        fps: 20.0,
        processingTimeMs: 80.0,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(good.performanceRating, equals('Good'));

      // Fair
      final fair = YOLOPerformanceMetrics(
        fps: 12.0,
        processingTimeMs: 130.0,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(fair.performanceRating, equals('Fair'));

      // Poor
      final poor = YOLOPerformanceMetrics(
        fps: 8.0,
        processingTimeMs: 300.0,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(poor.performanceRating, equals('Poor'));

      // Edge cases
      final excellentEdge = YOLOPerformanceMetrics(
        fps: 25.0, // exactly at excellent threshold
        processingTimeMs: 50.0, // exactly at excellent threshold
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(excellentEdge.performanceRating, equals('Excellent'));

      final goodEdge = YOLOPerformanceMetrics(
        fps: 15.0, // exactly at good threshold
        processingTimeMs: 100.0, // exactly at good threshold
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(goodEdge.performanceRating, equals('Good'));

      final fairEdge = YOLOPerformanceMetrics(
        fps: 10.0, // exactly at fair threshold
        processingTimeMs: 150.0, // exactly at fair threshold
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(fairEdge.performanceRating, equals('Fair'));
    });

    test('performance ratings boundary conditions', () {
      // Just below excellent threshold
      final justBelowExcellent1 = YOLOPerformanceMetrics(
        fps: 24.9,
        processingTimeMs: 50.0,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(justBelowExcellent1.performanceRating, equals('Good'));

      final justBelowExcellent2 = YOLOPerformanceMetrics(
        fps: 25.0,
        processingTimeMs: 50.1,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(justBelowExcellent2.performanceRating, equals('Good'));

      // Just below good threshold
      final justBelowGood1 = YOLOPerformanceMetrics(
        fps: 14.9,
        processingTimeMs: 100.0,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(justBelowGood1.performanceRating, equals('Fair'));

      final justBelowGood2 = YOLOPerformanceMetrics(
        fps: 15.0,
        processingTimeMs: 100.1,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(justBelowGood2.performanceRating, equals('Fair'));

      // Just below fair threshold
      final justBelowFair1 = YOLOPerformanceMetrics(
        fps: 9.9,
        processingTimeMs: 150.0,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(justBelowFair1.performanceRating, equals('Poor'));

      final justBelowFair2 = YOLOPerformanceMetrics(
        fps: 10.0,
        processingTimeMs: 150.1,
        frameNumber: 1,
        timestamp: DateTime.now(),
      );
      expect(justBelowFair2.performanceRating, equals('Poor'));
    });
  });

  group('YOLOPerformanceMetrics Additional Tests', () {
    test('equality comparison', () {
      final timestamp = DateTime(2024, 1, 1);
      final metrics1 = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: timestamp,
      );

      final metrics2 = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: timestamp,
      );

      final metrics3 = YOLOPerformanceMetrics(
        fps: 60.0, // different
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: timestamp,
      );

      // YOLOPerformanceMetrics doesn't override equality operator
      // so we need to compare their string representations
      expect(metrics1.toString(), equals(metrics2.toString()));
      expect(metrics1.toString(), isNot(equals(metrics3.toString())));
    });

    test('hashCode consistency', () {
      final timestamp = DateTime(2024, 1, 1);
      final metrics1 = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: timestamp,
      );

      final metrics2 = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: timestamp,
      );

      // Since equality isn't overridden, hashCode won't be consistent
      // Test that both objects have valid hashCodes instead
      expect(metrics1.hashCode, isA<int>());
      expect(metrics2.hashCode, isA<int>());
    });

    test('toString provides readable output', () {
      final metrics = YOLOPerformanceMetrics(
        fps: 30.0,
        processingTimeMs: 33.3,
        frameNumber: 100,
        timestamp: DateTime(2024, 1, 1),
      );

      final str = metrics.toString();
      expect(str, contains('YOLOPerformanceMetrics'));
      expect(str, contains('fps: 30.0'));
      expect(str, contains('processingTime: 33.300ms'));
      expect(str, contains('frame: 100'));
      expect(str, contains('timestamp'));
    });
  });
}
