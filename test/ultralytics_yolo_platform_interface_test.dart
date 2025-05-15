// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/ultralytics_yolo_platform_interface.dart';

class DummyPlatform extends UltralyticsYoloPlatform {}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('UltralyticsYoloPlatform', () {
    late DummyPlatform dummy;

    setUp(() {
      dummy = DummyPlatform();
    });

    group('Default Methods', () {
      group('Model Operations', () {
        test('should throw UnimplementedError for loadModel', () {
          expect(() => dummy.loadModel({}), throwsA(isA<UnimplementedError>()));
        });
      });

      group('Threshold Settings', () {
        test('should throw UnimplementedError for confidence threshold', () {
          expect(
            () => dummy.setConfidenceThreshold(0.5),
            throwsA(isA<UnimplementedError>()),
          );
        });

        test('should throw UnimplementedError for IOU threshold', () {
          expect(
            () => dummy.setIouThreshold(0.5),
            throwsA(isA<UnimplementedError>()),
          );
        });

        test('should throw UnimplementedError for num items threshold', () {
          expect(
            () => dummy.setNumItemsThreshold(1),
            throwsA(isA<UnimplementedError>()),
          );
        });
      });

      group('Camera Operations', () {
        test('should throw UnimplementedError for zoom ratio', () {
          expect(
            () => dummy.setZoomRatio(1),
            throwsA(isA<UnimplementedError>()),
          );
        });

        test('should throw UnimplementedError for lens direction', () {
          expect(
            () => dummy.setLensDirection(0),
            throwsA(isA<UnimplementedError>()),
          );
        });

        test('should throw UnimplementedError for camera operations', () {
          expect(dummy.closeCamera, throwsA(isA<UnimplementedError>()));
          expect(dummy.startCamera, throwsA(isA<UnimplementedError>()));
        });
      });

      group('Prediction Control', () {
        test('should throw UnimplementedError for prediction control', () {
          expect(dummy.pauseLivePrediction, throwsA(isA<UnimplementedError>()));
          expect(
            dummy.resumeLivePrediction,
            throwsA(isA<UnimplementedError>()),
          );
        });
      });

      group('Detection Operations', () {
        test('should throw UnimplementedError for detection operations', () {
          expect(
            () => dummy.detectionResultStream,
            throwsA(isA<UnimplementedError>()),
          );
          expect(
            () => dummy.detectImage(''),
            throwsA(isA<UnimplementedError>()),
          );
        });
      });

      group('Segmentation Operations', () {
        test('should throw UnimplementedError for segmentation operations', () {
          expect(
            () => dummy.segmentResultStream,
            throwsA(isA<UnimplementedError>()),
          );
          expect(
            () => dummy.segmentImage(''),
            throwsA(isA<UnimplementedError>()),
          );
        });
      });

      group('Classification Operations', () {
        test(
          'should throw UnimplementedError for classification operations',
          () {
            expect(
              () => dummy.classificationResultStream,
              throwsA(isA<UnimplementedError>()),
            );
            expect(
              () => dummy.classifyImage(''),
              throwsA(isA<UnimplementedError>()),
            );
          },
        );
      });

      group('Performance Metrics', () {
        test('should throw UnimplementedError for performance streams', () {
          expect(
            () => dummy.inferenceTimeStream,
            throwsA(isA<UnimplementedError>()),
          );
          expect(() => dummy.fpsRateStream, throwsA(isA<UnimplementedError>()));
        });
      });
    });

    group('Singleton Behavior', () {
      test('should allow setting and retrieving singleton instance', () {
        UltralyticsYoloPlatform.instance = dummy;
        expect(UltralyticsYoloPlatform.instance, dummy);
      });
    });

    group('Model Loading', () {
      test('should throw UnimplementedError for invalid model', () async {
        expect(
          () => dummy.loadModel({'invalid': 'model'}),
          throwsA(isA<UnimplementedError>()),
        );
      });
    });
  });
}
