// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'streaming_test_screen.dart';

void main() {
  runApp(const StreamingTestApp());
}

class StreamingTestApp extends StatelessWidget {
  const StreamingTestApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'YOLO Streaming Test',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.orange),
        useMaterial3: true,
      ),
      home: const StreamingTestScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}