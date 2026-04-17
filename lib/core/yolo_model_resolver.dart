// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';

import 'package:archive/archive.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';
import 'package:ultralytics_yolo/models/yolo_exceptions.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';

class _OfficialModelArtifact {
  const _OfficialModelArtifact({
    required this.id,
    required this.task,
    this.androidAssetName,
    this.iosArchiveName,
  });

  final String id;
  final YOLOTask task;
  final String? androidAssetName;
  final String? iosArchiveName;
}

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
  static bool get _isIosLikePlatform => Platform.isIOS || Platform.isMacOS;

  static const List<_OfficialModelArtifact> _officialModels = [
    _OfficialModelArtifact(
      id: 'yolo26n',
      task: YOLOTask.detect,
      androidAssetName: 'yolo26n_int8.tflite',
      iosArchiveName: 'yolo26n.mlpackage.zip',
    ),
    _OfficialModelArtifact(
      id: 'yolo26s',
      task: YOLOTask.detect,
      iosArchiveName: 'yolo26s.mlpackage.zip',
    ),
    _OfficialModelArtifact(
      id: 'yolo26m',
      task: YOLOTask.detect,
      iosArchiveName: 'yolo26m.mlpackage.zip',
    ),
    _OfficialModelArtifact(
      id: 'yolo26l',
      task: YOLOTask.detect,
      iosArchiveName: 'yolo26l.mlpackage.zip',
    ),
    _OfficialModelArtifact(
      id: 'yolo26x',
      task: YOLOTask.detect,
      iosArchiveName: 'yolo26x.mlpackage.zip',
    ),
    _OfficialModelArtifact(
      id: 'yolo26n-seg',
      task: YOLOTask.segment,
      androidAssetName: 'yolo26n-seg_int8.tflite',
    ),
    _OfficialModelArtifact(
      id: 'yolo26n-cls',
      task: YOLOTask.classify,
      androidAssetName: 'yolo26n-cls_int8.tflite',
    ),
    _OfficialModelArtifact(
      id: 'yolo26n-pose',
      task: YOLOTask.pose,
      androidAssetName: 'yolo26n-pose_int8.tflite',
    ),
    _OfficialModelArtifact(
      id: 'yolo26n-obb',
      task: YOLOTask.obb,
      androidAssetName: 'yolo26n-obb_int8.tflite',
    ),
    _OfficialModelArtifact(
      id: 'yolo11n',
      task: YOLOTask.detect,
      androidAssetName: 'yolo11n.tflite',
      iosArchiveName: 'yolo11n.mlpackage.zip',
    ),
    _OfficialModelArtifact(
      id: 'yolo11s',
      task: YOLOTask.detect,
      iosArchiveName: 'yolo11s.mlpackage.zip',
    ),
    _OfficialModelArtifact(
      id: 'yolo11m',
      task: YOLOTask.detect,
      iosArchiveName: 'yolo11m.mlpackage.zip',
    ),
    _OfficialModelArtifact(
      id: 'yolo11l',
      task: YOLOTask.detect,
      iosArchiveName: 'yolo11l.mlpackage.zip',
    ),
    _OfficialModelArtifact(
      id: 'yolo11x',
      task: YOLOTask.detect,
      iosArchiveName: 'yolo11x.mlpackage.zip',
    ),
    _OfficialModelArtifact(
      id: 'yolo11n-seg',
      task: YOLOTask.segment,
      androidAssetName: 'yolo11n-seg.tflite',
    ),
    _OfficialModelArtifact(
      id: 'yolo11n-cls',
      task: YOLOTask.classify,
      androidAssetName: 'yolo11n-cls.tflite',
    ),
    _OfficialModelArtifact(
      id: 'yolo11n-pose',
      task: YOLOTask.pose,
      androidAssetName: 'yolo11n-pose.tflite',
    ),
    _OfficialModelArtifact(
      id: 'yolo11n-obb',
      task: YOLOTask.obb,
      androidAssetName: 'yolo11n-obb.tflite',
    ),
  ];

  static List<String> officialModels({YOLOTask? task}) {
    return _officialModels
        .where((model) => task == null || model.task == task)
        .where(_isAvailableOnCurrentPlatform)
        .map((model) => model.id)
        .toList(growable: false);
  }

  static String? defaultOfficialModel({YOLOTask task = YOLOTask.detect}) {
    final models = officialModels(task: task);
    return models.isEmpty ? null : models.first;
  }

  static bool isOfficialModel(String source) =>
      _officialModelForId(_normalizeOfficialModelId(source)) != null;

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
    if (_officialModelForId(officialId) != null) {
      return _resolveOfficialModel(officialId!);
    }

    final uri = Uri.tryParse(source);
    if (uri != null && (uri.scheme == 'http' || uri.scheme == 'https')) {
      return _downloadRemoteModel(uri);
    }

    if (source.startsWith('assets/')) {
      return _isIosLikePlatform
          ? _resolveIosFlutterAsset(source)
          : _copyFlutterAssetToDocuments(source);
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

  static _OfficialModelArtifact? _officialModelForId(String? modelId) {
    if (modelId == null) return null;
    for (final model in _officialModels) {
      if (model.id == modelId) return model;
    }
    return null;
  }

  static bool _isAvailableOnCurrentPlatform(_OfficialModelArtifact model) {
    if (Platform.isAndroid) return model.androidAssetName != null;
    if (_isIosLikePlatform) return model.iosArchiveName != null;
    return model.androidAssetName != null || model.iosArchiveName != null;
  }

  static Future<String> _resolveOfficialModel(String modelId) async {
    final artifact = _officialModelForId(modelId);
    if (artifact == null) {
      throw ModelLoadingException('Unsupported official model: $modelId');
    }

    return _isIosLikePlatform
        ? _resolveIosOfficialModel(artifact)
        : _resolveAndroidOfficialModel(artifact);
  }

  static Future<String> _resolveAndroidOfficialModel(
    _OfficialModelArtifact artifact,
  ) async {
    final filename = artifact.androidAssetName;
    if (filename == null) {
      throw ModelLoadingException(
        'Official model ${artifact.id} is not available on Android.',
      );
    }
    final directory = await getApplicationDocumentsDirectory();
    final modelFile = File('${directory.path}/$filename');
    if (modelFile.existsSync()) return modelFile.path;

    if (await _copyFlutterAssetIfExists('assets/models/$filename', modelFile)) {
      return modelFile.path;
    }

    await _downloadToFile('$_latestReleaseBaseUrl/$filename', modelFile);
    return modelFile.path;
  }

  static Future<String> _resolveIosOfficialModel(
    _OfficialModelArtifact artifact,
  ) async {
    final archiveName = artifact.iosArchiveName;
    if (archiveName == null) {
      throw ModelLoadingException(
        'Official model ${artifact.id} is not available on iOS.',
      );
    }
    final directory = await getApplicationDocumentsDirectory();
    final modelDir = Directory('${directory.path}/${artifact.id}.mlpackage');
    if (await _hasValidMlPackage(modelDir)) return modelDir.path;
    if (modelDir.existsSync()) {
      modelDir.deleteSync(recursive: true);
    }

    final assetPath = 'assets/models/$archiveName';
    final assetBytes = await _loadAssetBytes(assetPath);
    if (assetBytes != null) {
      final extractedPath = await _extractMlPackageZip(assetBytes, modelDir);
      if (extractedPath != null) return extractedPath;
    }

    final archiveFile = File('${directory.path}/$archiveName');
    await _downloadToFile('$_latestReleaseBaseUrl/$archiveName', archiveFile);
    try {
      final bytes = archiveFile.readAsBytesSync();
      final extractedPath = await _extractMlPackageZip(bytes, modelDir);
      if (extractedPath == null) {
        throw ModelLoadingException('Failed to extract $archiveName.');
      }
      return extractedPath;
    } finally {
      if (archiveFile.existsSync()) {
        archiveFile.deleteSync();
      }
    }
  }

  static Future<String> _downloadRemoteModel(Uri uri) async {
    final documents = await getApplicationDocumentsDirectory();
    final fileName = uri.pathSegments.isEmpty ? 'model' : uri.pathSegments.last;

    if (_isIosLikePlatform && fileName.endsWith('.mlpackage.zip')) {
      final modelName = fileName.replaceAll('.mlpackage.zip', '');
      final targetDir = Directory('${documents.path}/$modelName.mlpackage');
      if (await _hasValidMlPackage(targetDir)) return targetDir.path;
      final archiveFile = File('${documents.path}/$fileName');
      await _downloadToFile(uri.toString(), archiveFile);
      try {
        final bytes = archiveFile.readAsBytesSync();
        final extractedPath = await _extractMlPackageZip(bytes, targetDir);
        if (extractedPath == null) {
          throw ModelLoadingException('Failed to extract $fileName.');
        }
        return extractedPath;
      } finally {
        if (archiveFile.existsSync()) {
          archiveFile.deleteSync();
        }
      }
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

  static Future<String> _resolveIosFlutterAsset(String assetPath) async {
    if (assetPath.endsWith('.mlpackage.zip')) {
      final fileName = assetPath.split('/').last;
      final modelName = fileName.replaceAll('.mlpackage.zip', '');
      final directory = await getApplicationDocumentsDirectory();
      final modelDir = Directory('${directory.path}/$modelName.mlpackage');
      if (await _hasValidMlPackage(modelDir)) return modelDir.path;

      final assetBytes = await _loadAssetBytes(assetPath);
      if (assetBytes == null) {
        throw ModelLoadingException('Flutter asset not found: $assetPath');
      }

      final extractedPath = await _extractMlPackageZip(assetBytes, modelDir);
      if (extractedPath == null) {
        throw ModelLoadingException('Failed to extract $assetPath.');
      }
      return extractedPath;
    }

    return assetPath;
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
    final temporaryFile = File('${targetFile.path}.download');
    try {
      if (temporaryFile.existsSync()) {
        temporaryFile.deleteSync();
      }
      final request = await client.getUrl(Uri.parse(url));
      final response = await request.close();
      if (response.statusCode != HttpStatus.ok) {
        throw ModelLoadingException(
          'Failed to download model from $url (HTTP ${response.statusCode}).',
        );
      }
      final sink = temporaryFile.openWrite();
      try {
        await response.pipe(sink);
      } finally {
        await sink.close();
      }
      if (targetFile.existsSync()) {
        targetFile.deleteSync();
      }
      temporaryFile.renameSync(targetFile.path);
    } catch (_) {
      if (temporaryFile.existsSync()) {
        temporaryFile.deleteSync();
      }
      rethrow;
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
        var relativePath = file.name.replaceAll('\\', '/');
        if (prefix != null) {
          if (relativePath.startsWith(prefix)) {
            relativePath = relativePath.substring(prefix.length);
          } else if (relativePath == prefix.replaceAll('/', '')) {
            continue;
          }
        }
        if (relativePath.isEmpty || !file.isFile) continue;
        final rootPath = targetDir.absolute.path;
        final rootPrefix = rootPath.endsWith(Platform.pathSeparator)
            ? rootPath
            : '$rootPath${Platform.pathSeparator}';
        final normalizedOutputPath = targetDir.uri
            .resolve(relativePath)
            .toFilePath();
        if (!normalizedOutputPath.startsWith(rootPrefix)) {
          throw ModelLoadingException(
            'Invalid archive entry outside target directory: $relativePath',
          );
        }
        final outputFile = File(normalizedOutputPath);
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
