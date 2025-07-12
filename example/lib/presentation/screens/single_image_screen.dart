// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'package:archive/archive_io.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/yolo.dart';

/// A screen that demonstrates YOLO inference on a single image.
///
/// This screen allows users to:
/// - Pick an image from the gallery
/// - Run YOLO inference on the selected image
/// - View detection results and annotated image
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

  YOLO? _yolo;
  bool _isModelReady = false;
  
  // Available models for different tasks
  final Map<YOLOTask, Map<String, String>> _availableModels = {
    YOLOTask.detect: {
      'name': 'yolo11n',
      'ios_package': 'yolo11n.mlpackage',
      'ios_zip': 'yolo11n.mlpackage.zip',
    },
    YOLOTask.segment: {
      'name': 'yolo11n-seg',
      'ios_package': 'yolo11n-seg.mlpackage',
      'ios_zip': 'yolo11n-seg.mlpackage.zip',
    },
    YOLOTask.classify: {
      'name': 'yolo11n-cls',
      'ios_package': 'yolo11n-cls.mlpackage',
      'ios_zip': 'yolo11n-cls.mlpackage.zip',
    },
    YOLOTask.pose: {
      'name': 'yolo11n-pose',
      'ios_package': 'yolo11n-pose.mlpackage',
      'ios_zip': 'yolo11n-pose.mlpackage.zip',
    },
    YOLOTask.obb: {
      'name': 'yolo11n-obb',
      'ios_package': 'yolo11n-obb.mlpackage',
      'ios_zip': 'yolo11n-obb.mlpackage.zip',
    },
  };
  
  // Current selected task
  YOLOTask _selectedTask = YOLOTask.detect;
  
  // Loading state for each task
  final Map<YOLOTask, bool> _modelLoadingStates = {
    YOLOTask.detect: false,
    YOLOTask.segment: false,
    YOLOTask.classify: false,
    YOLOTask.pose: false,
    YOLOTask.obb: false,
  };

  @override
  void initState() {
    super.initState();
    _initializeYOLO(_selectedTask);
  }
  
  @override
  void dispose() {
    _yolo?.dispose();
    super.dispose();
  }

  /// Initializes the YOLO model for inference
  ///
  /// For iOS:
  /// - Copies the .mlpackage from assets to local storage
  /// - Uses the local path for model loading
  /// For other platforms:
  /// - Uses the default asset path
  Future<void> _initializeYOLO(YOLOTask task) async {
    // Clean up previous model
    if (_yolo != null) {
      await _yolo!.dispose();
      _yolo = null;
    }
    
    setState(() {
      _isModelReady = false;
      _modelLoadingStates[task] = true;
    });
    
    final modelInfo = _availableModels[task]!;
    String modelPath = modelInfo['name']!;
    if (Platform.isIOS) {
      try {
        final localPath = await _copyMlPackageFromAssets(
          modelInfo['ios_package']!,
          modelInfo['ios_zip']!,
        );
        if (localPath != null) {
          modelPath = localPath;
          debugPrint('iOS: Using local .mlpackage path: $modelPath');
        } else {
          debugPrint(
            'iOS: Failed to copy .mlpackage, using default asset path.',
          );
        }
      } catch (e) {
        debugPrint('Error during .mlpackage copy for iOS: $e');
      }
    }

    _yolo = YOLO(modelPath: modelPath, task: task);

    try {
      await _yolo!.loadModel();
      if (mounted) {
        setState(() {
          _isModelReady = true;
          _modelLoadingStates[task] = false;
        });
      }
      debugPrint(
        'YOLO model initialized. Task: $task, Path: $modelPath, Ready: $_isModelReady',
      );
    } catch (e) {
      debugPrint('Error loading YOLO model: $e');
      if (mounted) {
        setState(() {
          _modelLoadingStates[task] = false;
        });
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error loading model: $e')));
      }
    }
  }

  /// Copies the .mlpackage from assets to local storage
  ///
  /// This is required for iOS to properly load the model.
  /// Returns the path to the local .mlpackage directory if successful,
  /// null otherwise.
  Future<String?> _copyMlPackageFromAssets(
    String mlPackageDirName,
    String mlPackageZipAssetName,
  ) async {
    try {
      final Directory appDocDir = await getApplicationDocumentsDirectory();
      final String localMlPackageDirPath =
          '${appDocDir.path}/$mlPackageDirName';
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

      final String assetZipPath = 'assets/models/$mlPackageZipAssetName';

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

        final String expectedPrefix = '$mlPackageDirName/';
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

  /// Picks an image from the gallery and runs inference
  ///
  /// This method:
  /// - Opens the image picker
  /// - Runs YOLO inference on the selected image
  /// - Updates the UI with detection results and annotated image
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
    final result = await _yolo!.predict(bytes);
    
    // Debug print to check what's returned
    print('DEBUG: Result keys: ${result.keys}');
    print('DEBUG: Has annotatedImage: ${result.containsKey('annotatedImage')}');
    if (result.containsKey('annotatedImage')) {
      print('DEBUG: annotatedImage type: ${result['annotatedImage'].runtimeType}');
      print('DEBUG: annotatedImage is Uint8List: ${result['annotatedImage'] is Uint8List}');
    }
    
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
          // Task selection dropdown
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Select Task:',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                DropdownButtonFormField<YOLOTask>(
                  value: _selectedTask,
                  decoration: InputDecoration(
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 12,
                    ),
                  ),
                  items: _availableModels.keys.map((task) {
                    return DropdownMenuItem(
                      value: task,
                      child: Row(
                        children: [
                          Text(_getTaskDisplayName(task)),
                          if (_modelLoadingStates[task]!)
                            const Padding(
                              padding: EdgeInsets.only(left: 8.0),
                              child: SizedBox(
                                width: 16,
                                height: 16,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                ),
                              ),
                            ),
                        ],
                      ),
                    );
                  }).toList(),
                  onChanged: (YOLOTask? newTask) {
                    if (newTask != null && newTask != _selectedTask) {
                      setState(() {
                        _selectedTask = newTask;
                        _detections = [];
                        _annotatedImage = null;
                        _imageBytes = null;
                      });
                      _initializeYOLO(newTask);
                    }
                  },
                ),
                const SizedBox(height: 8),
                Text(
                  'Model: ${_availableModels[_selectedTask]!['name']}',
                  style: const TextStyle(fontSize: 14, color: Colors.grey),
                ),
              ],
            ),
          ),
          const SizedBox(height: 10),
          ElevatedButton(
            onPressed: _isModelReady ? _pickAndPredict : null,
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
                  if (_detections.isNotEmpty) ...[
                    const Text(
                      'Detections:',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 8),
                    _buildDetectionsList(),
                  ],
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
  
  /// Returns a display name for each YOLO task
  String _getTaskDisplayName(YOLOTask task) {
    switch (task) {
      case YOLOTask.detect:
        return 'Object Detection';
      case YOLOTask.segment:
        return 'Instance Segmentation';
      case YOLOTask.classify:
        return 'Image Classification';
      case YOLOTask.pose:
        return 'Pose Estimation';
      case YOLOTask.obb:
        return 'Oriented Bounding Box';
    }
  }
  
  /// Builds a formatted list of detection results
  Widget _buildDetectionsList() {
    if (_selectedTask == YOLOTask.classify) {
      // For classification, show top predictions
      return Card(
        margin: const EdgeInsets.symmetric(horizontal: 16),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: _detections.map((detection) {
              final className = detection['cls'] ?? 'Unknown';
              final confidence = detection['conf'] ?? 0.0;
              return Padding(
                padding: const EdgeInsets.symmetric(vertical: 4.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(className, style: const TextStyle(fontSize: 14)),
                    Text(
                      '${(confidence * 100).toStringAsFixed(1)}%',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.bold,
                        color: Theme.of(context).primaryColor,
                      ),
                    ),
                  ],
                ),
              );
            }).toList(),
          ),
        ),
      );
    } else {
      // For other tasks, show detection count and details
      return Column(
        children: [
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 16),
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text('Total Detections:'),
                  Text(
                    '${_detections.length}',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 8),
          // Show first few detections
          ..._detections.take(5).map((detection) {
            final className = detection['cls'] ?? 'Unknown';
            final confidence = detection['conf'] ?? 0.0;
            return Card(
              margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              child: ListTile(
                title: Text(className),
                trailing: Text(
                  '${(confidence * 100).toStringAsFixed(1)}%',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Theme.of(context).primaryColor,
                  ),
                ),
              ),
            );
          }).toList(),
          if (_detections.length > 5)
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Text(
                '... and ${_detections.length - 5} more',
                style: const TextStyle(color: Colors.grey),
              ),
            ),
        ],
      );
    }
  }
}
