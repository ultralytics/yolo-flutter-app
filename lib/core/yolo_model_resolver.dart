// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';

import 'package:archive/archive.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';
import 'package:ultralytics_yolo/models/yolo_exceptions.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';

class YOLOResolvedModel {
  const YOLOResolvedModel({
    required this.modelPath,
    required this.task,
    required this.metadata,
  });

  final String modelPath;
  final YOLOTask task;
  final Map<String, dynamic> metadata;
}

class YOLOModelResolver {
  static const String _latestReleaseBaseUrl =
      'https://github.com/ultralytics/yolo-flutter-app/releases/latest/download';

  static final List<String> _officialModelIds = List.unmodifiable([
    for (final family in ['yolo26', 'yolo11'])
      for (final size in ['n', 's', 'm', 'l', 'x']) ...[
        '$family$size',
        '$family$size-seg',
        '$family$size-cls',
        '$family$size-pose',
        '$family$size-obb',
      ],
  ]);

  static List<String> officialModels({YOLOTask? task}) {
    if (task == null) return _officialModelIds;
    return _officialModelIds.where((id) {
      return switch (task) {
        YOLOTask.detect => !id.contains('-'),
        YOLOTask.segment => id.endsWith('-seg'),
        YOLOTask.classify => id.endsWith('-cls'),
        YOLOTask.pose => id.endsWith('-pose'),
        YOLOTask.obb => id.endsWith('-obb'),
      };
    }).toList();
  }

  static bool isOfficialModel(String source) =>
      _officialModelIds.contains(_normalizeOfficialModelId(source));

  static Future<YOLOResolvedModel> resolve({
    required String modelPath,
    YOLOTask? task,
  }) async {
    final resolvedPath = await preparePath(modelPath);
    final metadata = await inspect(resolvedPath);
    final metadataTask = YOLOTaskParsing.tryParse(metadata['task'] as String?);

    if (task != null && metadataTask != null && task != metadataTask) {
      throw ModelLoadingException(
        'Model task mismatch for $modelPath: expected ${task.name}, '
        'metadata says ${metadataTask.name}.',
      );
    }

    final effectiveTask = task ?? metadataTask;
    if (effectiveTask == null) {
      throw ModelLoadingException(
        'Could not determine the task for $modelPath. '
        'Provide task explicitly or use a model with exported metadata.',
      );
    }

    return YOLOResolvedModel(
      modelPath: resolvedPath,
      task: effectiveTask,
      metadata: metadata,
    );
  }

  static Future<Map<String, dynamic>> inspect(String modelPath) async {
    final channel = ChannelConfig.createSingleImageChannel();
    final result = await channel.invokeMethod('inspectModel', {
      'modelPath': modelPath,
    });
    if (result is! Map) return {};
    return Map<String, dynamic>.from(result);
  }

  static Future<String> preparePath(String modelPath) =>
      _resolvePath(modelPath);

  static Future<String> _resolvePath(String source) async {
    final officialId = _normalizeOfficialModelId(source);
    if (officialId != null && _officialModelIds.contains(officialId)) {
      return _resolveOfficialModel(officialId);
    }

    final uri = Uri.tryParse(source);
    if (uri != null && (uri.scheme == 'http' || uri.scheme == 'https')) {
      return _downloadRemoteModel(uri);
    }

    if (Platform.isAndroid && source.startsWith('assets/')) {
      return _copyFlutterAssetToDocuments(source);
    }

    return source;
  }

  static String? _normalizeOfficialModelId(String source) {
    final fileName = source.split('/').last;
    final normalized = fileName
        .replaceAll('.mlpackage.zip', '')
        .replaceAll('.mlpackage', '')
        .replaceAll('.mlmodelc', '')
        .replaceAll('.mlmodel', '')
        .replaceAll('.tflite', '');
    return normalized.isEmpty ? null : normalized;
  }

  static Future<String> _resolveOfficialModel(String modelId) async {
    return Platform.isIOS
        ? _resolveIosOfficialModel(modelId)
        : _resolveAndroidOfficialModel(modelId);
  }

  static Future<String> _resolveAndroidOfficialModel(String modelId) async {
    final filename = '$modelId.tflite';
    final directory = await getApplicationDocumentsDirectory();
    final modelFile = File('${directory.path}/$filename');
    if (modelFile.existsSync()) return modelFile.path;

    if (await _copyFlutterAssetIfExists('assets/models/$filename', modelFile)) {
      return modelFile.path;
    }

    await _downloadToFile('$_latestReleaseBaseUrl/$filename', modelFile);
    return modelFile.path;
  }

  static Future<String> _resolveIosOfficialModel(String modelId) async {
    final directory = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${directory.path}/$modelId.mlpackage');
    if (await _hasValidMlPackage(modelDir)) return modelDir.path;
    if (modelDir.existsSync()) {
      modelDir.deleteSync(recursive: true);
    }

    final assetPath = 'assets/models/$modelId.mlpackage.zip';
    final assetBytes = await _loadAssetBytes(assetPath);
    if (assetBytes != null) {
      final extractedPath = await _extractMlPackageZip(assetBytes, modelDir);
      if (extractedPath != null) return extractedPath;
    }

    final archiveFile = File('${directory.path}/$modelId.mlpackage.zip');
    await _downloadToFile(
      '$_latestReleaseBaseUrl/$modelId.mlpackage.zip',
      archiveFile,
    );
    final bytes = archiveFile.readAsBytesSync();
    final extractedPath = await _extractMlPackageZip(bytes, modelDir);
    if (extractedPath == null) {
      throw ModelLoadingException('Failed to extract $modelId.mlpackage.zip.');
    }
    return extractedPath;
  }

  static Future<String> _downloadRemoteModel(Uri uri) async {
    final documents = await getApplicationDocumentsDirectory();
    final fileName = uri.pathSegments.isEmpty ? 'model' : uri.pathSegments.last;

    if (Platform.isIOS && fileName.endsWith('.mlpackage.zip')) {
      final modelName = fileName.replaceAll('.mlpackage.zip', '');
      final targetDir = Directory('${documents.path}/$modelName.mlpackage');
      if (await _hasValidMlPackage(targetDir)) return targetDir.path;
      final archiveFile = File('${documents.path}/$fileName');
      await _downloadToFile(uri.toString(), archiveFile);
      final bytes = archiveFile.readAsBytesSync();
      final extractedPath = await _extractMlPackageZip(bytes, targetDir);
      if (extractedPath == null) {
        throw ModelLoadingException('Failed to extract $fileName.');
      }
      return extractedPath;
    }

    final file = File('${documents.path}/$fileName');
    if (file.existsSync()) return file.path;
    await _downloadToFile(uri.toString(), file);
    return file.path;
  }

  static Future<String> _copyFlutterAssetToDocuments(String assetPath) async {
    final fileName = assetPath.split('/').last;
    final directory = await getApplicationDocumentsDirectory();
    final file = File('${directory.path}/$fileName');
    if (file.existsSync()) return file.path;

    final assetBytes = await _loadAssetBytes(assetPath);
    if (assetBytes == null) {
      throw ModelLoadingException('Flutter asset not found: $assetPath');
    }

    file.writeAsBytesSync(assetBytes, flush: true);
    return file.path;
  }

  static Future<bool> _copyFlutterAssetIfExists(
    String assetPath,
    File targetFile,
  ) async {
    final assetBytes = await _loadAssetBytes(assetPath);
    if (assetBytes == null) return false;
    targetFile.parent.createSync(recursive: true);
    targetFile.writeAsBytesSync(assetBytes, flush: true);
    return true;
  }

  static Future<List<int>?> _loadAssetBytes(String assetPath) async {
    try {
      final asset = await rootBundle.load(assetPath);
      return asset.buffer.asUint8List();
    } catch (_) {
      return null;
    }
  }

  static Future<void> _downloadToFile(String url, File targetFile) async {
    targetFile.parent.createSync(recursive: true);
    final client = HttpClient();
    try {
      final request = await client.getUrl(Uri.parse(url));
      final response = await request.close();
      if (response.statusCode != HttpStatus.ok) {
        throw ModelLoadingException(
          'Failed to download model from $url (HTTP ${response.statusCode}).',
        );
      }
      final sink = targetFile.openWrite();
      await response.pipe(sink);
    } finally {
      client.close(force: true);
    }
  }

  static Future<bool> _hasValidMlPackage(Directory modelDir) async {
    return modelDir.existsSync() &&
        File('${modelDir.path}/Manifest.json').existsSync();
  }

  static Future<String?> _extractMlPackageZip(
    List<int> bytes,
    Directory targetDir,
  ) async {
    try {
      final archive = ZipDecoder().decodeBytes(bytes);
      if (targetDir.existsSync()) {
        targetDir.deleteSync(recursive: true);
      }
      targetDir.createSync(recursive: true);

      String? prefix;
      if (archive.files.isNotEmpty) {
        final first = archive.files.first.name;
        if (first.contains('/') &&
            first.split('/').first.endsWith('.mlpackage')) {
          final topLevelDir = first.split('/').first;
          if (archive.files.every(
            (file) =>
                file.name.startsWith('$topLevelDir/') ||
                file.name == topLevelDir,
          )) {
            prefix = '$topLevelDir/';
          }
        }
      }

      for (final file in archive) {
        var relativePath = file.name;
        if (prefix != null) {
          if (relativePath.startsWith(prefix)) {
            relativePath = relativePath.substring(prefix.length);
          } else if (relativePath == prefix.replaceAll('/', '')) {
            continue;
          }
        }
        if (relativePath.isEmpty || !file.isFile) continue;
        final outputFile = File('${targetDir.path}/$relativePath');
        outputFile.parent.createSync(recursive: true);
        outputFile.writeAsBytesSync(file.content as List<int>, flush: true);
      }

      return await _hasValidMlPackage(targetDir) ? targetDir.path : null;
    } catch (_) {
      if (targetDir.existsSync()) {
        targetDir.deleteSync(recursive: true);
      }
      return null;
    }
  }
}
