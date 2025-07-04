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

  /// Initializes the YOLO model for inference
  ///
  /// For iOS:
  /// - Copies the .mlpackage from assets to local storage
  /// - Uses the local path for model loading
  /// For other platforms:
  /// - Uses the default asset path
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

  /// Copies the .mlpackage from assets to local storage
  ///
  /// This is required for iOS to properly load the model.
  /// Returns the path to the local .mlpackage directory if successful,
  /// null otherwise.
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
