// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() => runApp(const MyApp());

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Example',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;
  List<YOLOResult>? _results;

  Future<void> detectObjects() async {
    // Pick an image
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile == null) return;

    setState(() {
      _image = File(pickedFile.path);
    });

    // Initialize YOLO
    final yolo = YOLO(modelPath: 'yolo11n.tflite');

    // Load model
    await yolo.loadModel();

    // Run inference
    final results = await yolo.predict(await _image!.readAsBytes());

    // Parse results
    setState(() {
      _results = (results['detections'] as List?)
          ?.map((e) => YOLOResult.fromMap(e))
          .toList();
    });

    // Clean up
    await yolo.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('YOLO Flutter Example')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            if (_image != null) Image.file(_image!, height: 300),
            if (_results != null) Text('Found ${_results!.length} objects'),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: detectObjects,
              child: const Text('Detect Objects'),
            ),
          ],
        ),
      ),
    );
  }
}
