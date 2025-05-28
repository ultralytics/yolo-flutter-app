// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// example/lib/main.dart
import 'dart:io'; // Added for File and Directory
import 'dart:typed_data';
import 'package:archive/archive_io.dart'; // Added for zip decoding
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle; // Added for rootBundle
import 'package:path_provider/path_provider.dart'; // Added for path_provider
import 'package:ultralytics_yolo/yolo.dart';
// YOLOResult is now imported through yolo.dart
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:image_picker/image_picker.dart';

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

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('YOLO Plugin Example')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const CameraInferenceScreen(),
                  ),
                );
              },
              child: const Text('Camera Inference'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const SingleImageScreen(),
                  ),
                );
              },
              child: const Text('Single Image Inference'),
            ),
          ],
        ),
      ),
    );
  }
}

class CameraInferenceScreen extends StatefulWidget {
  const CameraInferenceScreen({super.key});

  @override
  State<CameraInferenceScreen> createState() => _CameraInferenceScreenState();
}

class _CameraInferenceScreenState extends State<CameraInferenceScreen> {
  int _detectionCount = 0;
  double _confidenceThreshold = 0.5;
  double _iouThreshold = 0.45;
  int _numItemsThreshold = 30;
  String _lastDetection = "";
  double _currentProcessingTimeMs = 0.0;
  double _currentFps = 0.0;

  final _yoloController = YOLOViewController();
  final _yoloViewKey = GlobalKey<YOLOViewState>();
  bool _useController = true;

  void _onDetectionResults(List<YOLOResult> results) {
    if (!mounted) return;

    debugPrint('_onDetectionResults called with ${results.length} results');

    for (var i = 0; i < results.length && i < 3; i++) {
      final r = results[i];
      debugPrint(
        '  Detection $i: ${r.className} (${(r.confidence * 100).toStringAsFixed(1)}%) at ${r.boundingBox}',
      );
    }

    setState(() {
      _detectionCount = results.length;
      if (results.isNotEmpty) {
        final topDetection = results.reduce(
          (a, b) => a.confidence > b.confidence ? a : b,
        );
        _lastDetection =
            "${topDetection.className} (${(topDetection.confidence * 100).toStringAsFixed(1)}%)";
        debugPrint(
          'Updated state: count=$_detectionCount, top=$_lastDetection',
        );
      } else {
        _lastDetection = "None";
        debugPrint('Updated state: No detections');
      }
    });
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_useController) {
        _yoloController.setThresholds(
          confidenceThreshold: _confidenceThreshold,
          iouThreshold: _iouThreshold,
          numItemsThreshold: _numItemsThreshold,
        );
      } else {
        _yoloViewKey.currentState?.setThresholds(
          confidenceThreshold: _confidenceThreshold,
          iouThreshold: _iouThreshold,
          numItemsThreshold: _numItemsThreshold,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Camera Inference'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
        actions: [
          // Camera Switch Button
          IconButton(
            icon: const Icon(Icons.flip_camera_ios),
            tooltip: 'Switch Camera',
            onPressed: () {
              if (_useController) {
                _yoloController.switchCamera();
              } else {
                _yoloViewKey.currentState?.switchCamera();
              }
            },
          ),
          // Controller Toggle Button
          IconButton(
            icon: Icon(_useController ? Icons.gamepad : Icons.key),
            tooltip: _useController
                ? 'Using Controller'
                : 'Using Direct Access',
            onPressed: () {
              setState(() {
                _useController = !_useController;
              });
            },
          ),
        ],
      ),
      body: Column(
        children: [
          const SizedBox(height: 10),
          Container(
            padding: const EdgeInsets.all(8.0),
            // ignore: deprecated_member_use
            color: Colors.black.withOpacity(0.1),
            child: Column(
              // Changed to Column to stack rows of info
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text('Detection count: $_detectionCount'),
                    Text('Top detection: $_lastDetection'),
                  ],
                ),
                const SizedBox(height: 4), // Spacer
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Processing: ${_currentProcessingTimeMs.toStringAsFixed(0)}ms',
                    ),
                    Text('FPS: ${_currentFps.toStringAsFixed(1)}'),
                  ],
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Row(
              children: [
                const Text('Confidence threshold: '),
                Expanded(
                  child: Slider(
                    value: _confidenceThreshold,
                    min: 0.1,
                    max: 0.9,
                    divisions: 8,
                    label: _confidenceThreshold.toStringAsFixed(1),
                    onChanged: (value) {
                      setState(() {
                        _confidenceThreshold = value;
                        if (_useController) {
                          _yoloController.setConfidenceThreshold(value);
                        } else {
                          _yoloViewKey.currentState?.setConfidenceThreshold(
                            value,
                          );
                        }
                      });
                    },
                  ),
                ),
                Text('${(_confidenceThreshold * 100).toInt()}%'),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Row(
              children: [
                const Text('IoU threshold: '),
                Expanded(
                  child: Slider(
                    value: _iouThreshold,
                    min: 0.1,
                    max: 0.9,
                    divisions: 8,
                    label: _iouThreshold.toStringAsFixed(1),
                    onChanged: (value) {
                      setState(() {
                        _iouThreshold = value;
                        if (_useController) {
                          _yoloController.setIoUThreshold(value);
                        } else {
                          _yoloViewKey.currentState?.setIoUThreshold(value);
                        }
                      });
                    },
                  ),
                ),
                Text('${(_iouThreshold * 100).toInt()}%'),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Row(
              children: [
                const Text('Max detections: '),
                Expanded(
                  child: Slider(
                    value: _numItemsThreshold.toDouble(),
                    min: 5,
                    max: 50,
                    divisions: 9,
                    label: _numItemsThreshold.toString(),
                    onChanged: (value) {
                      setState(() {
                        _numItemsThreshold = value.toInt();
                        if (_useController) {
                          _yoloController.setNumItemsThreshold(
                            _numItemsThreshold,
                          );
                        } else {
                          _yoloViewKey.currentState?.setNumItemsThreshold(
                            _numItemsThreshold,
                          );
                        }
                      });
                    },
                  ),
                ),
                Text('$_numItemsThreshold'),
              ],
            ),
          ),
          Expanded(
            child: Container(
              color: Colors.black12,
              child: YOLOView(
                key: _useController ? null : _yoloViewKey,
                controller: _useController ? _yoloController : null,
                modelPath: 'yolo11n-seg', // Default model for camera view
                task: YOLOTask.segment,
                onResult: _onDetectionResults,
                onPerformanceMetrics: (metrics) {
                  if (mounted) {
                    setState(() {
                      _currentProcessingTimeMs =
                          metrics['processingTimeMs'] ?? 0.0;
                      _currentFps = metrics['fps'] ?? 0.0;
                    });
                    // Optional: print to debug console
                    // debugPrint('Perf Metrics: ${_currentProcessingTimeMs.toStringAsFixed(2)} ms, ${_currentFps.toStringAsFixed(1)} FPS');
                  }
                },
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class SingleImageScreen extends StatefulWidget {
  const SingleImageScreen({super.key});

  @override
  State<SingleImageScreen> createState() => _SingleImageScreenState();
}

class _SingleImageScreenState extends State<SingleImageScreen> {
  final _picker = ImagePicker();
  List<Map<String, dynamic>> _detections = [];
  Uint8List? _imageBytes;
  Uint8List? _annotatedImage;

  late YOLO _yolo;
  String _modelPathForYOLO =
      'yolo11n-seg'; // Default asset path for non-iOS or if local copy fails
  bool _isModelReady = false;

  // Name of the .mlpackage directory in local storage (after unzipping)
  final String _mlPackageDirName =
      'yolo11n-seg.mlpackage'; // Changed to yolo11n
  // Name of the zip file in assets (e.g., assets/models/yolo11n.mlpackage.zip)
  final String _mlPackageZipAssetName =
      'yolo11n-seg.mlpackage.zip'; // Changed to yolo11n

  @override
  void initState() {
    super.initState();
    _initializeYOLO();
  }

  Future<void> _initializeYOLO() async {
    if (Platform.isIOS) {
      try {
        final localPath = await _copyMlPackageFromAssets();
        if (localPath != null) {
          _modelPathForYOLO = localPath;
          debugPrint('iOS: Using local .mlpackage path: $_modelPathForYOLO');
        } else {
          debugPrint(
            'iOS: Failed to copy .mlpackage, using default asset path.',
          );
        }
      } catch (e) {
        debugPrint('Error during .mlpackage copy for iOS: $e');
      }
    }

    _yolo = YOLO(modelPath: _modelPathForYOLO, task: YOLOTask.segment);

    try {
      await _yolo.loadModel();
      if (mounted) {
        setState(() {
          _isModelReady = true;
        });
      }
      debugPrint(
        'YOLO model initialized. Path: $_modelPathForYOLO, Ready: $_isModelReady',
      );
    } catch (e) {
      debugPrint('Error loading YOLO model: $e');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error loading model: $e')));
      }
    }
  }

  Future<String?> _copyMlPackageFromAssets() async {
    try {
      final Directory appDocDir = await getApplicationDocumentsDirectory();
      final String localMlPackageDirPath =
          '${appDocDir.path}/$_mlPackageDirName';
      final Directory localMlPackageDir = Directory(localMlPackageDirPath);

      final manifestFile = File('$localMlPackageDirPath/Manifest.json');
      if (await localMlPackageDir.exists() && await manifestFile.exists()) {
        debugPrint(
          '.mlpackage directory and Manifest.json already exist and are correctly placed: $localMlPackageDirPath',
        );
        return localMlPackageDirPath;
      } else {
        if (await localMlPackageDir.exists()) {
          debugPrint(
            'Manifest.json not found at expected location or .mlpackage directory is incomplete. Will attempt to re-extract.',
          );
          // To ensure a clean state, you might consider deleting the directory first:
          // await localMlPackageDir.delete(recursive: true);
          // debugPrint('Deleted existing incomplete directory: $localMlPackageDirPath');
        }
        // Ensure the base directory exists before extraction
        if (!await localMlPackageDir.exists()) {
          await localMlPackageDir.create(recursive: true);
          debugPrint(
            'Created .mlpackage directory for extraction: $localMlPackageDirPath',
          );
        }
      }

      final String assetZipPath = 'assets/models/$_mlPackageZipAssetName';

      debugPrint(
        'Attempting to copy and unzip $assetZipPath to $localMlPackageDirPath',
      );

      final ByteData zipData = await rootBundle.load(assetZipPath);
      final List<int> zipBytes = zipData.buffer.asUint8List(
        zipData.offsetInBytes,
        zipData.lengthInBytes,
      );

      final archive = ZipDecoder().decodeBytes(zipBytes);

      for (final fileInArchive in archive) {
        final String originalFilenameInZip = fileInArchive.name;
        String filenameForExtraction = originalFilenameInZip;

        final String expectedPrefix = '$_mlPackageDirName/';
        if (originalFilenameInZip.startsWith(expectedPrefix)) {
          filenameForExtraction = originalFilenameInZip.substring(
            expectedPrefix.length,
          );
        }

        if (filenameForExtraction.isEmpty) {
          debugPrint(
            'Skipping empty filename after prefix strip: $originalFilenameInZip',
          );
          continue;
        }

        final filePath = '${localMlPackageDir.path}/$filenameForExtraction';

        if (fileInArchive.isFile) {
          final data = fileInArchive.content as List<int>;
          final localFile = File(filePath);
          try {
            await localFile.parent.create(recursive: true);
            await localFile.writeAsBytes(data);
            debugPrint(
              'Extracted file: $filePath (Size: ${data.length} bytes)',
            );
            if (filenameForExtraction == 'Manifest.json') {
              debugPrint('Manifest.json was written to $filePath');
            }
          } catch (e) {
            debugPrint('!!! Failed to write file $filePath: $e');
          }
        } else {
          final localDir = Directory(filePath);
          try {
            await localDir.create(recursive: true);
            debugPrint('Created directory: $filePath');
          } catch (e) {
            debugPrint('!!! Failed to create directory $filePath: $e');
          }
        }
      }

      final manifestFileAfterExtraction = File(
        '$localMlPackageDirPath/Manifest.json',
      );
      if (await manifestFileAfterExtraction.exists()) {
        debugPrint(
          'CONFIRMED: Manifest.json exists at ${manifestFileAfterExtraction.path}',
        );
      } else {
        debugPrint(
          'ERROR: Manifest.json DOES NOT exist at ${manifestFileAfterExtraction.path} after extraction loop.',
        );
      }

      debugPrint(
        'Successfully finished attempt to unzip .mlpackage to local storage: $localMlPackageDirPath',
      );
      return localMlPackageDirPath;
    } catch (e) {
      debugPrint('Error in _copyMlPackageFromAssets (outer try-catch): $e');
      return null;
    }
  }

  Future<void> _pickAndPredict() async {
    if (!_isModelReady) {
      debugPrint('Model not ready yet for inference.');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Model is loading, please wait...')),
        );
      }
      return;
    }
    final XFile? file = await _picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;

    final bytes = await file.readAsBytes();
    final result = await _yolo.predict(bytes);
    if (mounted) {
      setState(() {
        if (result.containsKey('boxes') && result['boxes'] is List) {
          _detections = List<Map<String, dynamic>>.from(result['boxes']);
        } else {
          _detections = [];
        }

        if (result.containsKey('annotatedImage') &&
            result['annotatedImage'] is Uint8List) {
          _annotatedImage = result['annotatedImage'] as Uint8List;
        } else {
          _annotatedImage = null;
        }

        _imageBytes = bytes;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Single Image Inference'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: Column(
        children: [
          const SizedBox(height: 20),
          ElevatedButton(
            onPressed: _pickAndPredict,
            child: const Text('Pick Image & Run Inference'),
          ),
          const SizedBox(height: 10),
          if (!_isModelReady && Platform.isIOS)
            const Padding(
              padding: EdgeInsets.all(8.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(width: 10),
                  Text("Preparing local model..."),
                ],
              ),
            )
          else if (!_isModelReady)
            const Padding(
              padding: EdgeInsets.all(8.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(width: 10),
                  Text("Model loading..."),
                ],
              ),
            ),
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                children: [
                  if (_annotatedImage != null)
                    SizedBox(
                      height: 300,
                      width: double.infinity,
                      child: Image.memory(_annotatedImage!),
                    )
                  else if (_imageBytes != null)
                    SizedBox(
                      height: 300,
                      width: double.infinity,
                      child: Image.memory(_imageBytes!),
                    ),
                  const SizedBox(height: 10),
                  const Text('Detections:'),
                  Text(_detections.toString()),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
