// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'package:archive/archive.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import '../models/model_type.dart';

/// Manages YOLO model loading, downloading, and caching.
class ModelManager {
  static const String _modelDownloadBaseUrl = 
      'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.0.0';
  
  /// Callback for download progress updates (0.0 to 1.0).
  final void Function(double progress)? onDownloadProgress;
  
  /// Callback for status message updates.
  final void Function(String message)? onStatusUpdate;
  
  ModelManager({
    this.onDownloadProgress,
    this.onStatusUpdate,
  });
  
  /// Get the appropriate model path for the current platform and model type.
  /// Returns null if model is not available and needs to be downloaded.
  Future<String?> getModelPath(ModelType modelType) async {
    if (Platform.isIOS) {
      return _getIOSModelPath(modelType);
    } else if (Platform.isAndroid) {
      return _getAndroidModelPath(modelType);
    }
    return null;
  }
  
  /// Get iOS model path (.mlpackage format).
  Future<String?> _getIOSModelPath(ModelType modelType) async {
    _updateStatus('Checking for ${modelType.modelName} model...');
    
    // Check if model exists in app bundle
    final bundledPath = '${modelType.modelName}.mlpackage';
    
    // For iOS, we need to check in the app's document directory
    final documentsDir = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${documentsDir.path}/${modelType.modelName}.mlpackage');
    
    if (await modelDir.exists()) {
      debugPrint('Found existing iOS model at: ${modelDir.path}');
      return modelDir.path;
    }
    
    // Try to download and extract
    _updateStatus('Downloading ${modelType.modelName} model...');
    
    final zipFile = File('${documentsDir.path}/${modelType.modelName}.mlpackage.zip');
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
      
      for (final file in archive) {
        final filename = file.name;
        final data = file.content as List<int>;
        final outputFile = File('${documentsDir.path}/$filename');
        await outputFile.create(recursive: true);
        await outputFile.writeAsBytes(data);
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
  
  /// Get Android model path (.tflite format).
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
    final modelFile = File('${documentsDir.path}/${modelType.modelName}.tflite');
    
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
  
  /// Clear all downloaded models from local storage.
  Future<void> clearCache() async {
    final documentsDir = await getApplicationDocumentsDirectory();
    
    // Clear all model files
    for (final modelType in ModelType.values) {
      // Android models
      final tfliteFile = File('${documentsDir.path}/${modelType.modelName}.tflite');
      if (await tfliteFile.exists()) {
        await tfliteFile.delete();
      }
      
      // iOS models
      final mlPackageDir = Directory('${documentsDir.path}/${modelType.modelName}.mlpackage');
      if (await mlPackageDir.exists()) {
        await mlPackageDir.delete(recursive: true);
      }
    }
  }
  
  /// Check if a model is available locally (either bundled or downloaded).
  Future<bool> isModelAvailable(ModelType modelType) async {
    final path = await getModelPath(modelType);
    return path != null;
  }
  
  void _updateStatus(String message) {
    debugPrint('ModelManager: $message');
    onStatusUpdate?.call(message);
  }
}