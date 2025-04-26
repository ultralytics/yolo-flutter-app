import 'dart:io' as io;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo/yolo_model.dart';
import 'package:ultralytics_yolo_example/image_picker.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ImagePickerScreen());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final controller = UltralyticsYoloCameraController();
  @override
  initState() {
    super.initState();
    _initSegmentDetectorWithLocalModel();
    // _initObjectClassifierWithLocalModel();
    // _initImageClassifierWithLocalModel();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: FutureBuilder<bool>(
          future: _checkPermissions(),
          builder: (context, snapshot) {
            final allPermissionsGranted = snapshot.data ?? false;
            print(allPermissionsGranted);
            return !allPermissionsGranted
                ? const Center(
                    child: Text("Error requesting permissions"),
                  )
                : FutureBuilder<SegmentDetector>(
                    future: _initSegmentDetectorWithLocalModel(),
                    builder: (context, snapshot) {
                      final predictor = snapshot.data;
                      return predictor == null
                          ? Container()
                          : Stack(
                              children: [
                                UltralyticsYoloCameraPreview(
                                  controller: controller,
                                  predictor: predictor,
                                  onCameraCreated: () {
                                    predictor.loadModel(useGpu: true);
                                  },
                                ),
                                StreamBuilder<List<DetectedSegment?>?>(
                                  stream: predictor.detectionResultStream,
                                  builder: (context, snapshot) {
                                    final detectionResults = snapshot.data;
                                    if (detectionResults != null &&
                                        detectionResults.isNotEmpty) {
                                      print(detectionResults.length);
                                      return CustomPaint(
                                        painter: Masks(
                                          polygons:
                                              detectionResults.first!.polygons,
                                          screenSize:
                                              MediaQuery.of(context).size,
                                          maskColor: Colors.red,
                                        ),
                                        size: MediaQuery.of(context).size,
                                      );
                                    } else {
                                      return const SizedBox();
                                    }
                                  },
                                ),
                                StreamBuilder<double?>(
                                  stream: predictor.inferenceTime,
                                  builder: (context, snapshot) {
                                    final inferenceTime = snapshot.data;

                                    return StreamBuilder<double?>(
                                      stream: predictor.fpsRate,
                                      builder: (context, snapshot) {
                                        final fpsRate = snapshot.data;

                                        return Times(
                                          inferenceTime: inferenceTime,
                                          fpsRate: fpsRate,
                                        );
                                      },
                                    );
                                  },
                                ),
                              ],
                            );
                    },
                  );
            // : FutureBuilder<ObjectClassifier>(
            //     future: _initObjectClassifierWithLocalModel(),
            //     builder: (context, snapshot) {
            //       final predictor = snapshot.data;

            //       return predictor == null
            //           ? Container()
            //           : Stack(
            //               children: [
            //                 UltralyticsYoloCameraPreview(
            //                   controller: controller,
            //                   predictor: predictor,
            //                   onCameraCreated: () {
            //                     predictor.loadModel();
            //                   },
            //                 ),
            //                 StreamBuilder<double?>(
            //                   stream: predictor.inferenceTime,
            //                   builder: (context, snapshot) {
            //                     final inferenceTime = snapshot.data;

            //                     return StreamBuilder<double?>(
            //                       stream: predictor.fpsRate,
            //                       builder: (context, snapshot) {
            //                         final fpsRate = snapshot.data;

            //                         return Times(
            //                           inferenceTime: inferenceTime,
            //                           fpsRate: fpsRate,
            //                         );
            //                       },
            //                     );
            //                   },
            //                 ),
            //               ],
            //             );
            //     },
            //   );
          },
        ),
        floatingActionButton: FloatingActionButton(
          child: const Icon(Icons.cameraswitch),
          onPressed: () {
            controller.toggleLensDirection();
          },
        ),
      ),
    );
  }

  // Future<ObjectDetector> _initObjectDetectorWithLocalModel() async {
  //   //   // FOR IOS
  //   //   final modelPath = await _copy('assets/yolov8n.mlmodel');
  //   //   final model = LocalYoloModel(
  //   //     id: '',
  //   //     task: Task.detect,
  //   //     format: Format.coreml,
  //   //     modelPath: modelPath,
  //   //   );
  //   // FOR ANDROID
  //   final modelPath = await _copy('assets/yolov8n_int8.tflite');
  //   final metadataPath = await _copy('assets/obj.yaml');
  //   final model = LocalYoloModel(
  //     id: '',
  //     task: Task.detect,
  //     format: Format.tflite,
  //     modelPath: modelPath,
  //     metadataPath: metadataPath,
  //   );

  //   return ObjectDetector(model: model);
  // }

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

  // Future<ImageClassifier> _initImageClassifierWithLocalModel() async {
  //   final modelPath = await _copy('assets/yolov8n-cls.mlmodel');
  //   final model = LocalYoloModel(
  //     id: '',
  //     task: Task.classify,
  //     format: Format.coreml,
  //     modelPath: modelPath,
  //   );

  //   // final modelPath = await _copy('assets/yolov8n-cls.bin');
  //   // final paramPath = await _copy('assets/yolov8n-cls.param');
  //   // final metadataPath = await _copy('assets/metadata-cls.yaml');
  //   // final model = LocalYoloModel(
  //   //   id: '',
  //   //   task: Task.classify,
  //   //   modelPath: modelPath,
  //   //   paramPath: paramPath,
  //   //   metadataPath: metadataPath,
  //   // );

  //   return ImageClassifier(model: model);
  // }

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

  Future<bool> _checkPermissions() async {
    List<Permission> permissions = [];

    var cameraStatus = await Permission.camera.status;
    if (!cameraStatus.isGranted) permissions.add(Permission.camera);

    // var storageStatus = await Permission.photos.status;
    // if (!storageStatus.isGranted) permissions.add(Permission.photos);

    if (permissions.isEmpty) {
      return true;
    } else {
      try {
        Map<Permission, PermissionStatus> statuses =
            await permissions.request();
        return statuses.values
            .every((status) => status == PermissionStatus.granted);
      } on Exception catch (_) {
        return false;
      }
    }
  }
}

class Times extends StatelessWidget {
  const Times({
    super.key,
    required this.inferenceTime,
    required this.fpsRate,
  });

  final double? inferenceTime;
  final double? fpsRate;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Align(
        alignment: Alignment.bottomCenter,
        child: Container(
            margin: const EdgeInsets.all(20),
            padding: const EdgeInsets.all(20),
            decoration: const BoxDecoration(
              borderRadius: BorderRadius.all(Radius.circular(10)),
              color: Colors.black54,
            ),
            child: Text(
              '${(inferenceTime ?? 0).toStringAsFixed(1)} ms  -  ${(fpsRate ?? 0).toStringAsFixed(1)} FPS',
              style: const TextStyle(color: Colors.white70),
            )),
      ),
    );
  }
}

class Masks extends CustomPainter {
  Masks({
    required this.polygons,
    required this.screenSize,
    required this.maskColor,
  });

  final List<dynamic> polygons;
  final Size screenSize;
  final Color maskColor;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = maskColor.withOpacity(0.5) // Adjust opacity as needed
      ..style = PaintingStyle.fill;

    for (final polygonList in polygons) {
      if (polygonList is List<dynamic>) {
        final path = Path();
        bool firstPoint = true;

        for (final pointData in polygonList) {
          if (pointData is Map<String, dynamic> &&
              pointData.containsKey('x') &&
              pointData.containsKey('y')) {
            final double xRatio = (pointData['x'] as num).toDouble();
            final double yRatio = (pointData['y'] as num).toDouble();

            // Scale the normalized coordinates to the actual screen size
            final double x = xRatio * screenSize.width;
            final double y = yRatio * screenSize.height;

            if (firstPoint) {
              path.moveTo(x, y);
              firstPoint = false;
            } else {
              path.lineTo(x, y);
            }
          } else if (pointData is Offset) {
            // Handle direct Offset objects if that's a possibility
            if (firstPoint) {
              path.moveTo(pointData.dx, pointData.dy);
              firstPoint = false;
            } else {
              path.lineTo(pointData.dx, pointData.dy);
            }
          }
        }

        path.close();
        canvas.drawPath(path, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant Masks oldDelegate) {
    return oldDelegate.polygons != polygons ||
        oldDelegate.screenSize != screenSize ||
        oldDelegate.maskColor != maskColor;
  }
}
