import 'dart:io' as io;
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/predict/segment/detected_segment.dart';
import 'package:ultralytics_yolo/predict/segment/segment_detector.dart';
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
    print(assetPath);
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
    print("here lkanfbknaob");
    print(metadataPath);
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
      final response = await predictor.loadModel(useGpu: true);
      predictor.setConfidenceThreshold(0.7);
      final result = await predictor.detect(imagePath: image.path);
      setState(
        () {
          if (result != null) {
            for (var seg in result) {
              if (seg != null) {
                box.add(seg);
                segments.addAll(seg.polygons);
              }
            }
          }
        },
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: SafeArea(
        child: Scaffold(
          body: Column(
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
                          CustomPaint(
                            painter: Masks(
                              polygons: segments,
                              imageSize: imageSize,
                              maskColor: Colors.red,
                              displayWidth: MediaQuery.of(context).size.width,
                            ),
                            size: imageSize,
                          ),
                          CustomPaint(
                            painter: BoundingBox(
                              boxes: box,
                              imageSize: imageSize,
                              displayWidth: MediaQuery.of(context).size.width,
                            ),
                            size: imageSize,
                          ),
                        ],
                      ),
                    )
                  : const SizedBox(),
              Center(
                child: ElevatedButton(
                  onPressed: () => _pickImage(),
                  child: Text("pick image"),
                ),
              )
            ],
          ),
        ),
      ),
    );
  }
}

class Masks extends CustomPainter {
  Masks({
    required this.polygons,
    required this.imageSize,
    required this.maskColor,
    required this.displayWidth,
  });

  final List<List<Offset>> polygons;
  final Size imageSize; // Actual image size
  final Color maskColor;
  final double displayWidth;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = maskColor.withOpacity(0.5)
      ..style = PaintingStyle.fill;

    // scale from 160x160 to actual image size
    final double displayHeight =
        (imageSize.height / imageSize.width) * displayWidth;
    final double scaleX = displayWidth / 80.0;
    final double scaleY = displayHeight / 80.0;

    for (final polygon in polygons) {
      if (polygon.isNotEmpty) {
        final path = Path();

        final scaledPoints = polygon.map((point) {
          return Offset(point.dx * scaleX, point.dy * scaleY);
        }).toList();

        path.addPolygon(scaledPoints, true);
        canvas.drawPath(path, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant Masks oldDelegate) {
    return oldDelegate.polygons != polygons ||
        oldDelegate.imageSize != imageSize ||
        oldDelegate.maskColor != maskColor;
  }
}

class BoundingBox extends CustomPainter {
  BoundingBox({
    required this.boxes,
    required this.imageSize,
    required this.displayWidth,
  });

  final List<DetectedSegment> boxes;
  final Size imageSize;
  final double displayWidth;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.blue.withOpacity(0.5)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;

    final double scaleX = displayWidth / imageSize.width;
    final double displayHeight =
        (imageSize.height / imageSize.width) * displayWidth;
    final double scaleY = displayHeight / imageSize.height;

    for (final DetectedSegment box in boxes) {
      canvas.drawRect(
        Rect.fromLTWH(
          box.boundingBox.left * displayWidth,
          box.boundingBox.top * displayHeight,
          box.boundingBox.width * displayWidth,
          box.boundingBox.height * displayHeight,
        ),
        paint,
      );

      final textSpan = TextSpan(
        text: box.label,
        style: const TextStyle(
          color: Colors.black,
          fontSize: 16,
          fontWeight: FontWeight.bold,
        ),
      );

      final textPainter = TextPainter(
        text: textSpan,
        textAlign: TextAlign.left,
        textDirection: TextDirection.ltr,
      );

      textPainter.layout(
        minWidth: 0,
        maxWidth: size.width,
      );

      final offset = Offset(
        box.boundingBox.left * displayWidth,
        box.boundingBox.top * displayHeight,
      );

      textPainter.paint(canvas, offset);
    }
  }

  @override
  bool shouldRepaint(covariant BoundingBox oldDelegate) {
    return oldDelegate.boxes != boxes;
  }
}
