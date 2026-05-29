// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Real-time YOLO camera inference. Thin shell over [YOLOShowcase] that wires the capture callback to the platform
/// share sheet via `share_plus`. Kept intentionally bare so the screen reads side-by-side with `yolo-ios-app`'s
/// `ViewController` — no extra Material chrome on top.
class CameraInferenceScreen extends StatelessWidget {
  const CameraInferenceScreen({super.key});

  Future<void> _onCapture(Uint8List bytes) async {
    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/yolo_capture.jpg')..writeAsBytesSync(bytes);
    await SharePlus.instance.share(
      ShareParams(files: [XFile(file.path)], text: 'Ultralytics YOLO'),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Hide the Semantic tab to mirror the iOS app's task control (Detect / Segment / Classify / Pose / OBB).
    return Scaffold(
      body: YOLOShowcase(onCapture: _onCapture, showSemanticTask: false),
    );
  }
}
