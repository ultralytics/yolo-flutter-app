import 'package:flutter/foundation.dart';

void logInfo(String message) {
  if (kDebugMode) {
    print('[YOLO DEBUG] $message');
  }
}
