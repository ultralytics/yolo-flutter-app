// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter_test/flutter_test.dart';
import 'package:ultralytics_yolo/yolo_exceptions.dart';

void main() {
  group('YOLOException', () {
    test('constructor creates instance with message', () {
      const message = 'Test error message';
      final exception = YOLOException(message);

      expect(exception.message, message);
    });

    test('toString returns formatted message', () {
      const message = 'Test error message';
      final exception = YOLOException(message);

      expect(exception.toString(), 'YOLOException: $message');
    });
  });

  group('ModelLoadingException', () {
    test('constructor creates instance with message', () {
      const message = 'Failed to load model';
      final exception = ModelLoadingException(message);

      expect(exception.message, message);
    });

    test('toString returns formatted message', () {
      const message = 'Failed to load model';
      final exception = ModelLoadingException(message);

      expect(exception.toString(), 'ModelLoadingException: $message');
    });
  });

  group('ModelNotLoadedException', () {
    test('constructor creates instance with message', () {
      const message = 'Model not loaded';
      final exception = ModelNotLoadedException(message);

      expect(exception.message, message);
    });

    test('toString returns formatted message', () {
      const message = 'Model not loaded';
      final exception = ModelNotLoadedException(message);

      expect(exception.toString(), 'ModelNotLoadedException: $message');
    });
  });
}
