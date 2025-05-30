// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'package:archive/archive.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import '../models/model_type.dart';

/// Manages YOLO model loading, downloading, and caching.
///
/// This class handles:
/// - Checking for existing models in the app bundle
/// - Downloading models from the Ultralytics GitHub releases
/// - Extracting and caching models locally
/// - Platform-specific model path management
class ModelManager {
  /// Base URL for downloading model files from GitHub releases
  static const String _modelDownloadBaseUrl =
      'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.0.0';

  static const MethodChannel _channel = MethodChannel(
    'yolo_single_image_channel',
  );

  /// Callback for download progress updates (0.0 to 1.0)
  final void Function(double progress)? onDownloadProgress;

  /// Callback for status message updates
  final void Function(String message)? onStatusUpdate;

  /// Creates a new ModelManager instance
  ///
  /// [onDownloadProgress] is called with progress updates during model downloads
  /// [onStatusUpdate] is called with status messages during model operations
  ModelManager({this.onDownloadProgress, this.onStatusUpdate});

  /// Gets the appropriate model path for the current platform and model type.
  /// For iOS: Always downloads model if not found locally to avoid crashes
  /// For Android: Checks bundled assets first, then downloaded models.
  ///
  /// Returns the path to the model file if it exists locally, or null if the model
  /// needs to be downloaded. The path format depends on the platform:
  /// - iOS: Path to .mlpackage directory
  /// - Android: Path to .tflite file
  Future<String?> getModelPath(ModelType modelType) async {
    if (Platform.isIOS) {
      return _getIOSModelPath(modelType);
    } else if (Platform.isAndroid) {
      return _getAndroidModelPath(modelType);
    }
    return null;
  }

  /// Check if a model exists in the iOS bundle (Xcode project).
  /// This is useful to verify if a model is bundled before using it.
  Future<bool> isModelBundled(ModelType modelType) async {
    if (!Platform.isIOS) {
      return false;
    }

    final bundleCheck = await _checkModelExistsInBundle(modelType.modelName);
    return bundleCheck['exists'] == true;
  }

  /// Check if a model exists locally (downloaded).
  Future<bool> isModelDownloaded(ModelType modelType) async {
    if (Platform.isIOS) {
      final documentsDir = await getApplicationDocumentsDirectory();
      final modelDir = Directory(
        '${documentsDir.path}/${modelType.modelName}.mlpackage',
      );
      return modelDir.exists();
    } else if (Platform.isAndroid) {
      final documentsDir = await getApplicationDocumentsDirectory();
      final modelFile = File(
        '${documentsDir.path}/${modelType.modelName}.tflite',
      );
      return modelFile.exists();
    }
    return false;
  }

  /// Download model if not available locally.
  Future<String?> downloadModelIfNeeded(ModelType modelType) async {
    if (Platform.isIOS) {
      return _downloadIOSModel(modelType);
    }
    // Android download is handled in getAndroidModelPath
    return null;
  }

  /// Gets the iOS model path (.mlpackage format).
  /// This method checks in the following order:
  /// 1. Bundle models (if exists, returns model name)
  /// 2. Downloaded models (returns full path)
  /// 3. Downloads the model if not found
  ///
  /// Checks for the model in the app's documents directory and downloads it if needed.
  /// Returns the path to the .mlpackage directory if successful, null otherwise.
  Future<String?> _getIOSModelPath(ModelType modelType) async {
    _updateStatus('Checking for ${modelType.modelName} model...');

    // Step 1: Check if model exists in iOS bundle
    try {
      final bundleCheck = await _checkModelExistsInBundle(modelType.modelName);
      if (bundleCheck['exists'] == true) {
        debugPrint(
          'Found bundled iOS model: ${modelType.modelName} at ${bundleCheck['location']}',
        );
        // For bundled models, return just the model name
        // The native code will resolve the actual path
        return modelType.modelName;
      }
    } catch (e) {
      debugPrint('Error checking bundle for model: $e');
    }

    // Step 2: Check downloaded models
    final documentsDir = await getApplicationDocumentsDirectory();
    final modelDir = Directory(
      '${documentsDir.path}/${modelType.modelName}.mlpackage',
    );

    if (await modelDir.exists()) {
      // Verify it's a valid mlpackage by checking for Manifest.json
      final manifestFile = File('${modelDir.path}/Manifest.json');
      if (await manifestFile.exists()) {
        debugPrint('Found downloaded iOS model at: ${modelDir.path}');
        return modelDir.path;
      } else {
        debugPrint(
          'Invalid mlpackage directory (missing Manifest.json), removing...',
        );
        await modelDir.delete(recursive: true);
      }
    }

    // Step 3: Model not found anywhere, download it
    debugPrint('Model not found locally or in bundle, downloading...');
    _updateStatus('Downloading ${modelType.modelName} model...');
    return _downloadIOSModel(modelType);
  }

  /// Check if a model exists in the iOS bundle
  Future<Map<String, dynamic>> _checkModelExistsInBundle(
    String modelName,
  ) async {
    if (!Platform.isIOS) {
      return {'exists': false};
    }

    try {
      final result = await _channel.invokeMethod('checkModelExists', {
        'modelPath': modelName,
      });
      return Map<String, dynamic>.from(result);
    } catch (e) {
      debugPrint('Error checking model in bundle: $e');
      return {'exists': false};
    }
  }

  /// Download iOS model (.mlpackage format).
  Future<String?> _downloadIOSModel(ModelType modelType) async {
    final documentsDir = await getApplicationDocumentsDirectory();
    final modelDir = Directory(
      '${documentsDir.path}/${modelType.modelName}.mlpackage',
    );

    // If already exists, return it
    if (await modelDir.exists()) {
      return modelDir.path;
    }

    _updateStatus('Downloading ${modelType.modelName} model...');

    final zipFile = File(
      '${documentsDir.path}/${modelType.modelName}.mlpackage.zip',
    );
    final url = '$_modelDownloadBaseUrl/${modelType.modelName}.mlpackage.zip';

    try {
      final client = http.Client();
      final request = await client.send(http.Request('GET', Uri.parse(url)));
      final contentLength = request.contentLength ?? 0;

      // Download with progress tracking
      final bytes = <int>[];
      int downloadedBytes = 0;

      await for (final chunk in request.stream) {
        bytes.addAll(chunk);
        downloadedBytes += chunk.length;

        if (contentLength > 0) {
          final progress = downloadedBytes / contentLength;
          onDownloadProgress?.call(progress);
        }
      }

      await zipFile.writeAsBytes(bytes);
      client.close();

      // Extract the zip file
      _updateStatus('Extracting model...');
      final archive = ZipDecoder().decodeBytes(await zipFile.readAsBytes());

      // Check if the archive has a redundant top-level directory
      // This happens when the mlpackage directory itself was zipped
      bool hasRedundantDirectory = false;
      String? redundantDirPrefix;

      // Check if all files start with the same directory name
      if (archive.files.isNotEmpty) {
        final firstPath = archive.files.first.name;
        if (firstPath.contains('/')) {
          final topDir = firstPath.split('/').first;
          // Check if it's a redundant mlpackage directory
          if (topDir.endsWith('.mlpackage')) {
            // Check if ALL files start with this directory
            hasRedundantDirectory = archive.files.every(
              (f) => f.name.startsWith('$topDir/') || f.name == topDir,
            );
            if (hasRedundantDirectory) {
              redundantDirPrefix = '$topDir/';
              debugPrint('Detected redundant directory structure: $topDir');
            }
          }
        }
      }

      // Create the mlpackage directory first
      await modelDir.create(recursive: true);
      debugPrint('Created mlpackage directory: ${modelDir.path}');

      for (final file in archive) {
        String filename = file.name;

        // Remove redundant directory prefix if present
        if (hasRedundantDirectory && redundantDirPrefix != null) {
          if (filename.startsWith(redundantDirPrefix)) {
            filename = filename.substring(redundantDirPrefix.length);
          } else if (filename == redundantDirPrefix.replaceAll('/', '')) {
            // Skip the directory entry itself
            continue;
          }
        }

        // Skip empty filenames
        if (filename.isEmpty) continue;

        if (file.isFile) {
          final data = file.content as List<int>;
          // Extract files into the mlpackage directory
          final outputFile = File('${modelDir.path}/$filename');
          await outputFile.parent.create(recursive: true);
          await outputFile.writeAsBytes(data);
          debugPrint('Extracted: ${outputFile.path}');
        } else if (!hasRedundantDirectory) {
          // Only create directories if we're not stripping a redundant prefix
          final dir = Directory('${modelDir.path}/$filename');
          await dir.create(recursive: true);
          debugPrint('Created directory: ${dir.path}');
        }
      }

      // Clean up zip file
      await zipFile.delete();

      // Verify extraction
      if (await modelDir.exists()) {
        return modelDir.path;
      } else {
        debugPrint('Error: mlpackage directory not found after extraction');
      }
    } catch (e) {
      debugPrint('Failed to download/extract iOS model: $e');
      if (await zipFile.exists()) {
        await zipFile.delete();
      }
      // Clean up the model directory if extraction failed
      if (await modelDir.exists()) {
        await modelDir.delete(recursive: true);
      }
    }

    return null;
  }

  /// Gets the Android model path (.tflite format)
  ///
  /// Checks for the model in the app's assets and local storage, and downloads it if needed.
  /// Returns the path to the .tflite file if successful, null otherwise.
  Future<String?> _getAndroidModelPath(ModelType modelType) async {
    _updateStatus('Checking for ${modelType.modelName} model...');

    // First check if model exists in assets (bundled)
    final bundledModelName = '${modelType.modelName}.tflite';

    try {
      // Try to load from assets
      await rootBundle.load('assets/models/$bundledModelName');
      debugPrint('Using bundled Android model: $bundledModelName');
      return bundledModelName;
    } catch (e) {
      // Model not in assets, continue to check local storage
      debugPrint('Model not found in assets, checking local storage...');
    }

    // Check if model exists in local storage (previously downloaded)
    final documentsDir = await getApplicationDocumentsDirectory();
    final modelFile = File(
      '${documentsDir.path}/${modelType.modelName}.tflite',
    );

    if (await modelFile.exists()) {
      debugPrint('Found existing Android model at: ${modelFile.path}');
      return modelFile.path;
    }

    // Download model from GitHub
    _updateStatus('Downloading ${modelType.modelName} model...');

    final url = '$_modelDownloadBaseUrl/${modelType.modelName}.tflite';

    try {
      final client = http.Client();
      final request = await client.send(http.Request('GET', Uri.parse(url)));
      final contentLength = request.contentLength ?? 0;

      // Download with progress tracking
      final bytes = <int>[];
      int downloadedBytes = 0;

      await for (final chunk in request.stream) {
        bytes.addAll(chunk);
        downloadedBytes += chunk.length;

        if (contentLength > 0) {
          final progress = downloadedBytes / contentLength;
          onDownloadProgress?.call(progress);
        }
      }

      client.close();

      if (bytes.isNotEmpty) {
        await modelFile.writeAsBytes(bytes);
        return modelFile.path;
      }
    } catch (e) {
      debugPrint('Failed to download Android model: $e');
    }

    return null;
  }

  /// Clears all downloaded models from local storage
  ///
  /// This removes both iOS (.mlpackage) and Android (.tflite) models
  /// from the app's documents directory.
  Future<void> clearCache() async {
    _updateStatus('Clearing model cache...');
    final documentsDir = await getApplicationDocumentsDirectory();

    // Clear all model files
    for (final modelType in ModelType.values) {
      // Android models
      final tfliteFile = File(
        '${documentsDir.path}/${modelType.modelName}.tflite',
      );
      if (await tfliteFile.exists()) {
        await tfliteFile.delete();
        debugPrint(
          'Deleted cached Android model: ${modelType.modelName}.tflite',
        );
      }

      // iOS models
      final mlPackageDir = Directory(
        '${documentsDir.path}/${modelType.modelName}.mlpackage',
      );
      if (await mlPackageDir.exists()) {
        await mlPackageDir.delete(recursive: true);
        debugPrint(
          'Deleted cached iOS model: ${modelType.modelName}.mlpackage',
        );
      }
    }

    debugPrint('Model cache cleared successfully');
    _updateStatus('Cache cleared');
  }

  /// Force re-download a specific model (useful for debugging).
  Future<String?> forceDownloadModel(ModelType modelType) async {
    _updateStatus('Force downloading ${modelType.modelName} model...');

    if (Platform.isIOS) {
      // Delete existing model if any
      final documentsDir = await getApplicationDocumentsDirectory();
      final modelDir = Directory(
        '${documentsDir.path}/${modelType.modelName}.mlpackage',
      );
      if (await modelDir.exists()) {
        await modelDir.delete(recursive: true);
        debugPrint('Deleted existing model before re-download');
      }

      return _downloadIOSModel(modelType);
    } else if (Platform.isAndroid) {
      // Delete existing model if any
      final documentsDir = await getApplicationDocumentsDirectory();
      final modelFile = File(
        '${documentsDir.path}/${modelType.modelName}.tflite',
      );
      if (await modelFile.exists()) {
        await modelFile.delete();
        debugPrint('Deleted existing model before re-download');
      }

      // Re-download by calling the existing download logic
      _updateStatus('Downloading ${modelType.modelName} model...');
      final url = '$_modelDownloadBaseUrl/${modelType.modelName}.tflite';

      try {
        final client = http.Client();
        final request = await client.send(http.Request('GET', Uri.parse(url)));
        final contentLength = request.contentLength ?? 0;

        // Download with progress tracking
        final bytes = <int>[];
        int downloadedBytes = 0;

        await for (final chunk in request.stream) {
          bytes.addAll(chunk);
          downloadedBytes += chunk.length;

          if (contentLength > 0) {
            final progress = downloadedBytes / contentLength;
            onDownloadProgress?.call(progress);
          }
        }

        client.close();

        if (bytes.isNotEmpty) {
          await modelFile.writeAsBytes(bytes);
          return modelFile.path;
        }
      } catch (e) {
        debugPrint('Failed to download Android model: $e');
      }
    }

    return null;
  }

  /// Checks if a model is available locally (either bundled or downloaded)
  ///
  /// Returns true if the model exists and is ready to use, false otherwise.
  Future<bool> isModelAvailable(ModelType modelType) async {
    final path = await getModelPath(modelType);
    return path != null;
  }

  /// Updates the status message and logs it
  void _updateStatus(String message) {
    debugPrint('ModelManager: $message');
    onStatusUpdate?.call(message);
  }
}
