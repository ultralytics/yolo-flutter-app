import 'dart:io' as io;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo/yolo_model.dart';

class ImagePickerScreen extends StatefulWidget {
  const ImagePickerScreen({super.key});

  @override
  State<ImagePickerScreen> createState() => _ImagePickerState();
}

class _ImagePickerState extends State<ImagePickerScreen> {
  List<List<Offset>> segments = [];
  String imagePath = '';
  Size imageSize = const Size(0, 0);
  List<DetectedSegment> box = [];

  Future<String> _copy(String assetPath) async {
    final path = '${(await getApplicationSupportDirectory()).path}/$assetPath';
    await io.Directory(dirname(path)).create(recursive: true);
    final file = io.File(path);
    if (!await file.exists()) {
      final byteData = await rootBundle.load(assetPath);
      await file.writeAsBytes(byteData.buffer
          .asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));
    }
    return file.path;
  }

  Future<SegmentDetector> _initSegmentDetectorWithLocalModel() async {
    final modelPath = await _copy('assets/yolo11n-seg_float16.tflite');
    final metadataPath = await _copy('assets/metaxy.yaml');
    final model = LocalYoloModel(
      id: '',
      task: Task.segment,
      format: Format.tflite,
      modelPath: modelPath,
      metadataPath: metadataPath,
    );

    return SegmentDetector(model: model);
  }

  Future<void> _pickImage() async {
    segments.clear();
    box.clear();

    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      imagePath = image.path;
      final bytes = await io.File(image.path).readAsBytes();
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      final imageGet = frame.image;
      imageSize = Size(imageGet.width.toDouble(), imageGet.height.toDouble());
      final predictor = await _initSegmentDetectorWithLocalModel();
      await predictor.loadModel(useGpu: true);
      predictor.setConfidenceThreshold(0.7);
      final result = await predictor.detect(imagePath: image.path);

      if (result != null) {
        for (var seg in result) {
          if (seg != null) {
            box.add(seg);

            segments.addAll(seg.polygons);
          }
        }
      }
    }
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: SafeArea(
        child: Scaffold(
          body: Padding(
            padding: const EdgeInsets.symmetric(
              vertical: 16.0,
            ),
            child: Column(
              children: [
                segments.isNotEmpty && imagePath.isNotEmpty
                    ? Expanded(
                        child: Stack(
                          children: [
                            Image.file(
                              io.File(
                                imagePath,
                              ),
                              width: double.infinity,
                              fit: BoxFit.cover,
                            ),
                            RepaintBoundary(
                              child: CustomPaint(
                                painter: SegmentDetectorPainter(
                                  results: box,
                                  imageSize: imageSize,
                                  displayWidth:
                                      MediaQuery.of(context).size.width,
                                ),
                                size: imageSize,
                              ),
                            ),
                          ],
                        ),
                      )
                    : const SizedBox(),
                Expanded(
                  child: Center(
                    child: ElevatedButton(
                      onPressed: () => _pickImage(),
                      child: const Text("pick image"),
                    ),
                  ),
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
}
