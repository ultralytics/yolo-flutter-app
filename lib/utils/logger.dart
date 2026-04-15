// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/foundation.dart';

void logInfo(String message) {
  if (kDebugMode) {
    debugPrint('[YOLO DEBUG] $message');
  }
}
