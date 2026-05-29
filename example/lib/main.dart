// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:flutter_native_splash/flutter_native_splash.dart';
import 'package:ultralytics_yolo_example/presentation/screens/camera_inference_screen.dart';
import 'package:ultralytics_yolo_example/presentation/screens/single_image_screen.dart';

void main() {
  // Hold the native splash (window-level, so it also covers the camera platform view) until the live view is ready.
  // CameraInferenceScreen removes it via YOLOShowcase.onReady once the first inference result arrives.
  final binding = WidgetsFlutterBinding.ensureInitialized();
  FlutterNativeSplash.preserve(widgetsBinding: binding);
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ultralytics YOLO',
      themeMode: ThemeMode.dark,
      theme: ThemeData.dark(useMaterial3: true),
      initialRoute: '/',
      routes: {
        '/': (_) => const CameraInferenceScreen(),
        '/single': (_) => const SingleImageScreen(),
      },
    );
  }
}
