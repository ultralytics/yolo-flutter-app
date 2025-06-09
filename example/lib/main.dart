// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// example/lib/main.dart
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo_example/presentation/screens/camera_inference_screen.dart';

void main() {
  runApp(const YOLOExampleApp());
}

class YOLOExampleApp extends StatelessWidget {
  const YOLOExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(title: 'YOLO Plugin Example', home: HomeScreen());
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool showCamera = false;
  int navigationCount = 0;
  Key cameraKey = UniqueKey();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('YOLO Test - Navigations: $navigationCount'),
      ),
      body: showCamera
          ? CameraInferenceScreen(key: cameraKey)
          : Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    'Camera disposal test\nNavigations: $navigationCount',
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.headlineMedium,
                  ),
                  const SizedBox(height: 32),
                  ElevatedButton(
                    onPressed: () {
                      setState(() {
                        showCamera = true;
                        navigationCount++;
                        cameraKey = UniqueKey(); // Force widget recreation
                      });
                    },
                    child: const Text('Open Camera'),
                  ),
                ],
              ),
            ),
      floatingActionButton: showCamera
          ? FloatingActionButton(
              onPressed: () {
                print('🏠 Home button pressed - disposing camera widget');
                setState(() {
                  showCamera = false;
                });
                print('🏠 setState completed - camera widget should be disposed');
              },
              child: const Icon(Icons.home),
            )
          : null,
    );
  }
}
