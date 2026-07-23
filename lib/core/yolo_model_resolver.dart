// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/config/channel_config.dart';
import 'package:ultralytics_yolo/core/yolo_model_manager.dart';
import 'package:ultralytics_yolo/models/yolo_exceptions.dart';
import 'package:ultralytics_yolo/models/yolo_task.dart';
import 'package:ultralytics_yolo/utils/mini_zip.dart';

class _OfficialModelArtifact {
  const _OfficialModelArtifact({
    required this.id,
    required this.task,
    required this.androidAssetName,
    required this.iosArchiveName,
  });

  final String id;
  final YOLOTask task;
  final String androidAssetName;
  final String iosArchiveName;
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
  // Pinned release assets provide reproducible first-use downloads. Update these constants, docs, and URL tests together
  // when the official model asset set moves to a new release. LiteRT and opt-in QNN assets share the Android model
  // release; QNN models are referenced by explicit paths rather than model-ID resolution.
  static const String _androidModelReleaseBaseUrl =
      'https://github.com/ultralytics/yolo-flutter-app/releases/download/models-v1.0.0';
  static const String _iosModelReleaseBaseUrl =
      'https://github.com/ultralytics/yolo-ios-app/releases/download/models-v1.0.0';
  static bool get _isIosLikePlatform => Platform.isIOS || Platform.isMacOS;

  static const List<String> _yolo26Sizes = ['n', 's', 'm', 'l', 'x'];
  // Canonical YOLO26 task x size matrix. Keep generated so the app, docs, and export script all represent the same
  // 7-task x 5-size official asset set. YOLO11 assets still exist on older releases but are no longer maintained for
  // autodownload — load them as custom paths/URLs instead.
  static final List<_OfficialModelArtifact> _officialModels = [
    for (final task in YOLOTask.values)
      for (final size in _yolo26Sizes) _yolo26Artifact(task: task, size: size),
  ];

  static _OfficialModelArtifact _yolo26Artifact({
    required YOLOTask task,
    required String size,
  }) {
    final id = 'yolo26$size${task.modelSuffix}';
    return _OfficialModelArtifact(
      id: id,
      task: task,
      androidAssetName: '${id}_w8a32.tflite',
      iosArchiveName: '$id.mlpackage.zip',
    );
  }

  static List<String> officialModels({YOLOTask? task}) {
    return _officialModels
        .where((model) => task == null || model.task == task)
        .map((model) => model.id)
        .toList(growable: false);
  }

  static String? defaultOfficialModel({YOLOTask task = YOLOTask.detect}) {
    final models = officialModels(task: task);
    return models.isEmpty ? null : models.first;
  }

  static bool isOfficialModel(String source) =>
      _officialModelForId(_normalizeOfficialModelId(source)) != null;

  @visibleForTesting
  static String? officialModelDownloadUrlForTesting(
    String modelId, {
    required bool iosLike,
  }) {
    final artifact = _officialModelForId(modelId);
    if (artifact == null) return null;
    return iosLike
        ? '$_iosModelReleaseBaseUrl/${artifact.iosArchiveName}'
        : '$_androidModelReleaseBaseUrl/${artifact.androidAssetName}';
  }

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
    final uri = Uri.tryParse(source);
    if (uri != null && (uri.scheme == 'http' || uri.scheme == 'https')) {
      return _downloadRemoteModel(uri);
    }

    if (source.startsWith('assets/')) {
      return _isIosLikePlatform
          ? _resolveIosFlutterAsset(source)
          : _copyFlutterAssetToDocuments(source);
    }

    if (_officialModelForId(source) != null) {
      return _resolveOfficialModel(source);
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

  static Future<String> _resolveOfficialModel(String modelId) async {
    final artifact = _officialModelForId(modelId);
    if (artifact == null) {
      throw ModelLoadingException('Unsupported official model: $modelId');
    }

    return _isIosLikePlatform
        ? _resolveIosOfficialModel(artifact)
        : _resolveAndroidOfficialModel(artifact);
  }

  /// Whether official [modelId] is already available without a network download.
  static Future<bool> isOfficialModelAvailableLocally(String modelId) async {
    final artifact = _officialModelForId(modelId);
    if (artifact == null) return false;
    final directory = await getApplicationDocumentsDirectory();
    if (_isIosLikePlatform) {
      if (await _hasValidMlPackage(
        Directory('${directory.path}/${artifact.id}.mlpackage'),
      )) {
        return true;
      }
      return await _loadAssetBytes(
            'assets/models/${artifact.iosArchiveName}',
          ) !=
          null;
    }
    final filename = artifact.androidAssetName;
    if (File('${directory.path}/$filename').existsSync()) return true;
    return await _loadAssetBytes('assets/models/$filename') != null;
  }

  static Future<String> _resolveAndroidOfficialModel(
    _OfficialModelArtifact artifact,
  ) async {
    final filename = artifact.androidAssetName;
    final directory = await getApplicationDocumentsDirectory();
    final modelFile = File('${directory.path}/$filename');
    if (modelFile.existsSync()) return modelFile.path;

    if (await _copyFlutterAssetIfExists('assets/models/$filename', modelFile)) {
      return modelFile.path;
    }

    await _downloadToFile(
      '$_androidModelReleaseBaseUrl/$filename',
      modelFile,
      progressId: artifact.id,
    );
    return modelFile.path;
  }

  static Future<String> _resolveIosOfficialModel(
    _OfficialModelArtifact artifact,
  ) async {
    final archiveName = artifact.iosArchiveName;
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
    await _downloadToFile(
      '$_iosModelReleaseBaseUrl/$archiveName',
      archiveFile,
      progressId: artifact.id,
    );
    return _extractMlPackageArchiveFile(archiveFile, archiveName, modelDir);
  }

  static Future<String> _downloadRemoteModel(Uri uri) async {
    final documents = await getApplicationDocumentsDirectory();
    final fileName = uri.pathSegments.isEmpty ? 'model' : uri.pathSegments.last;

    if (_isIosLikePlatform && fileName.endsWith('.mlpackage.zip')) {
      final modelName = fileName.replaceAll('.mlpackage.zip', '');
      final targetDir = Directory('${documents.path}/$modelName.mlpackage');
      if (await _hasValidMlPackage(targetDir)) return targetDir.path;
      final archiveFile = File('${documents.path}/$fileName');
      await _downloadToFile(uri.toString(), archiveFile, progressId: modelName);
      return _extractMlPackageArchiveFile(archiveFile, fileName, targetDir);
    }

    final file = File('${documents.path}/$fileName');
    if (file.existsSync()) return file.path;
    await _downloadToFile(
      uri.toString(),
      file,
      progressId: _normalizeOfficialModelId(fileName),
    );
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

  static Future<void> _downloadToFile(
    String url,
    File targetFile, {
    String? progressId,
  }) async {
    targetFile.parent.createSync(recursive: true);
    final client = HttpClient();
    Object? downloadToken;
    if (progressId != null) {
      downloadToken = YOLOModelManager.registerDownload(
        progressId,
        () => client.close(force: true),
      );
    }
    final temporaryFile = File('${targetFile.path}.download');
    void checkCancelled() {
      if (progressId != null &&
          downloadToken != null &&
          YOLOModelManager.isDownloadCancelled(progressId, downloadToken)) {
        throw ModelLoadingException('Model download canceled.');
      }
    }

    try {
      if (temporaryFile.existsSync()) {
        temporaryFile.deleteSync();
      }
      checkCancelled();
      final request = await client.getUrl(Uri.parse(url));
      final response = await request.close();
      checkCancelled();
      if (response.statusCode != HttpStatus.ok) {
        throw ModelLoadingException(
          'Failed to download model from $url (HTTP ${response.statusCode}).',
        );
      }

      // Stream bytes to disk so we can tally `received / contentLength` and surface progress through
      // `YOLOModelManager.emitProgress`. `pipe` would block progress reporting until completion.
      final totalBytes = response.contentLength;
      var receivedBytes = 0;
      double lastFraction = -1;

      if (progressId != null) {
        YOLOModelManager.emitProgress(progressId, 0);
      }

      final sink = temporaryFile.openWrite();
      try {
        await response.forEach((chunk) {
          checkCancelled();
          sink.add(chunk);
          if (progressId == null || totalBytes <= 0) return;
          receivedBytes += chunk.length;
          // Cap the in-flight fraction at 0.99 so listeners never observe `1.0` from the streaming loop — the terminal
          // emit at `1.0` is reserved for the post-rename success path so a chip never lights up "downloaded" for a
          // transfer that turns out to be 0-byte / corrupt.
          final fraction = (receivedBytes / totalBytes).clamp(0.0, 0.99);
          // Throttle to ~1% steps to avoid flooding the stream on fast links.
          if (fraction - lastFraction >= 0.01) {
            lastFraction = fraction;
            YOLOModelManager.emitProgress(progressId, fraction);
          }
        });
      } finally {
        await sink.close();
      }

      checkCancelled();
      // Some endpoints (e.g. GitHub release redirects that resolve to a 200 with no body, or chunked-encoding
      // responses that completed with zero chunks) leave the `.download` file in a state where openWrite + close
      // never materialised a real file on disk. Fall through to renameSync would then throw `PathNotFoundException`
      // with errno=2, which is confusing for users — surface a clean ModelLoadingException instead. We deliberately
      // delay the terminal `emitProgress(1)` until AFTER the file has been validated and renamed so a listener that
      // marks the chip "downloaded" never sees success for a failed transfer.
      if (!temporaryFile.existsSync() || temporaryFile.lengthSync() == 0) {
        throw ModelLoadingException(
          'Downloaded 0 bytes for $url. The asset may be missing from the release.',
        );
      }

      if (targetFile.existsSync()) {
        targetFile.deleteSync();
      }
      temporaryFile.renameSync(targetFile.path);

      if (progressId != null) {
        YOLOModelManager.emitProgress(progressId, 1);
      }
    } catch (_) {
      if (temporaryFile.existsSync()) {
        temporaryFile.deleteSync();
      }
      if (progressId != null &&
          downloadToken != null &&
          YOLOModelManager.isDownloadCancelled(progressId, downloadToken)) {
        throw ModelLoadingException('Model download canceled.');
      }
      rethrow;
    } finally {
      if (progressId != null && downloadToken != null) {
        YOLOModelManager.finishDownload(progressId, downloadToken);
      }
      client.close(force: true);
    }
  }

  static Future<bool> _hasValidMlPackage(Directory modelDir) async {
    return modelDir.existsSync() &&
        File('${modelDir.path}/Manifest.json').existsSync();
  }

  static Future<String> _extractMlPackageArchiveFile(
    File archiveFile,
    String displayName,
    Directory targetDir,
  ) async {
    try {
      final extractedPath = await _extractMlPackageZip(
        archiveFile.readAsBytesSync(),
        targetDir,
      );
      if (extractedPath == null) {
        throw ModelLoadingException('Failed to extract $displayName.');
      }
      return extractedPath;
    } finally {
      if (archiveFile.existsSync()) {
        archiveFile.deleteSync();
      }
    }
  }

  static Future<String?> _extractMlPackageZip(
    List<int> bytes,
    Directory targetDir,
  ) async {
    try {
      if (targetDir.existsSync()) {
        targetDir.deleteSync(recursive: true);
      }
      targetDir.createSync(recursive: true);

      MiniZip.extractBytes(
        bytes,
        destination: targetDir,
        stripTopLevelDirectoryEndingWith: '.mlpackage',
        skip: (path) => path.startsWith('__MACOSX/') || path.contains('/._'),
      );

      return await _hasValidMlPackage(targetDir) ? targetDir.path : null;
    } catch (_) {
      if (targetDir.existsSync()) {
        targetDir.deleteSync(recursive: true);
      }
      return null;
    }
  }
}
