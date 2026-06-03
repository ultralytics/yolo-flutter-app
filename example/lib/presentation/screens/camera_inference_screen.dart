// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Real-time YOLO camera inference. Thin shell over [YOLOShowcase] that wires the capture callback to the platform
/// share sheet via `share_plus` and stamps the live app version in the bottom-left (matching `yolo-ios-app`). Kept
/// intentionally bare so the screen reads side-by-side with `yolo-ios-app`'s `ViewController` — no extra Material
/// chrome on top.
class CameraInferenceScreen extends StatefulWidget {
  const CameraInferenceScreen({super.key});

  @override
  State<CameraInferenceScreen> createState() => _CameraInferenceScreenState();
}

class _CameraInferenceScreenState extends State<CameraInferenceScreen> {
  String? _versionLabel;

  @override
  void initState() {
    super.initState();
    _loadVersion();
  }

  Future<void> _loadVersion() async {
    final info = await PackageInfo.fromPlatform();
    if (!mounted) return;
    // Match yolo-ios-app's `labelVersion`: "v<version> (<build>)".
    setState(() => _versionLabel = 'v${info.version} (${info.buildNumber})');
  }

  Future<void> _onCapture(Uint8List bytes) async {
    // Capture the share-sheet anchor BEFORE any async gap (no BuildContext use after await). iOS 26 gives the activity
    // controller a popoverPresentationController even on iPhone; without a valid source rect the popover anchors at
    // (0,0) and can present/dismiss incorrectly (and it is required on iPad). Use the screen's render box.
    final box = context.findRenderObject() as RenderBox?;
    final origin = box != null && box.hasSize
        ? box.localToGlobal(Offset.zero) & box.size
        : null;

    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/yolo_capture.jpg')..writeAsBytesSync(bytes);

    await SharePlus.instance.share(
      ShareParams(
        files: [XFile(file.path)],
        text: 'Ultralytics YOLO',
        sharePositionOrigin: origin,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    // Show all 6 tasks (Detect / Segment / Semantic / Classify / Pose / OBB) to match the iOS app's task control.
    return Scaffold(
      body: YOLOShowcase(versionLabel: _versionLabel, onCapture: _onCapture),
    );
  }
}
