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
  });

  final String id;
  final YOLOTask task;
  final String androidAssetName;
}

/// Apple model formats. Both are directories that ship zipped and are validated by a marker file after extraction.
enum _AppleModelFormat {
  coreML('.mlpackage', 'Manifest.json'),
  coreAI('.aimodel', 'metadata.json');

  const _AppleModelFormat(this.suffix, this.markerFile);

  final String suffix;
  final String markerFile;

  String get archiveSuffix => '$suffix.zip';

  bool isValid(Directory modelDir) =>
      File('${modelDir.path}/$markerFile').existsSync();

  static _AppleModelFormat? ofArchive(String fileName) {
    for (final format in values) {
      if (fileName.endsWith(format.archiveSuffix)) return format;
    }
    return null;
  }
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
  // when the official model asset set moves to a new release. The official Android assets are LiteRT `_w8a32.tflite`
  // and opt-in QNN `_qnn.onnx` models on v0.6.6. QNN models use explicit paths rather than model-ID resolution. The
  // iOS release hosts every model as both Core AI `.aimodel.zip` (iOS 27+) and Core ML `.mlpackage.zip`.
  static const String _androidModelReleaseBaseUrl =
      'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.6.6';
  static const String _iosModelReleaseBaseUrl =
      'https://github.com/ultralytics/yolo-ios-app/releases/download/v8.3.0';
  static const String _officialModelCacheDirectory = 'mobile-standard-v1';
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
    bool coreAI = false,
  }) {
    final artifact = _officialModelForId(modelId);
    if (artifact == null) return null;
    final format = coreAI ? _AppleModelFormat.coreAI : _AppleModelFormat.coreML;
    return iosLike
        ? '$_iosModelReleaseBaseUrl/${artifact.id}${format.archiveSuffix}'
        : '$_androidModelReleaseBaseUrl/${artifact.androidAssetName}';
  }

  /// Core AI (`.aimodel`) is the default on iOS 27 and later; Core ML (`.mlpackage`) remains the fallback for earlier
  /// iOS versions and for the iOS Simulator, which does not ship Core AI. Only the native side can tell them apart.
  static Future<_AppleModelFormat> _preferredAppleFormat() async {
    final available = await ChannelConfig.createSingleImageChannel()
        .invokeMethod<bool>('isCoreAIAvailable');
    return available == true
        ? _AppleModelFormat.coreAI
        : _AppleModelFormat.coreML;
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
        .replaceAll('.aimodel.zip', '')
        .replaceAll('.aimodel', '')
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
    final documents = await getApplicationDocumentsDirectory();
    final directory = Directory(
      '${documents.path}/$_officialModelCacheDirectory',
    );
    if (_isIosLikePlatform) {
      final preferred = await _preferredAppleFormat();
      if (preferred.isValid(
        Directory('${directory.path}/${artifact.id}${preferred.suffix}'),
      )) {
        return true;
      }
      for (final format in {preferred, _AppleModelFormat.coreML}) {
        final assetPath = 'assets/models/${artifact.id}${format.archiveSuffix}';
        if (await _loadAssetBytes(assetPath) != null) return true;
      }
      return false;
    }
    final filename = artifact.androidAssetName;
    if (File('${directory.path}/$filename').existsSync()) return true;
    return await _loadAssetBytes('assets/models/$filename') != null;
  }

  static Future<String> _resolveAndroidOfficialModel(
    _OfficialModelArtifact artifact,
  ) async {
    final filename = artifact.androidAssetName;
    final documents = await getApplicationDocumentsDirectory();
    final directory = Directory(
      '${documents.path}/$_officialModelCacheDirectory',
    );
    final modelFile = File('${directory.path}/$filename');
    if (modelFile.existsSync()) return modelFile.path;
    final legacyFile = File('${documents.path}/$filename');
    if (legacyFile.existsSync()) legacyFile.deleteSync();

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
    final preferred = await _preferredAppleFormat();
    final documents = await getApplicationDocumentsDirectory();
    final directory = Directory(
      '${documents.path}/$_officialModelCacheDirectory',
    );
    final preferredDir = Directory(
      '${directory.path}/${artifact.id}${preferred.suffix}',
    );
    if (preferred.isValid(preferredDir)) return preferredDir.path;
    final legacyModelDir = Directory(
      '${documents.path}/${artifact.id}.mlpackage',
    );
    if (legacyModelDir.existsSync()) {
      legacyModelDir.deleteSync(recursive: true);
    }

    // Bundled assets win over downloads, so an app that bundles only Core ML stays offline on iOS 27.
    for (final format in {preferred, _AppleModelFormat.coreML}) {
      final assetBytes = await _loadAssetBytes(
        'assets/models/${artifact.id}${format.archiveSuffix}',
      );
      if (assetBytes == null) continue;
      final modelDir = Directory(
        '${directory.path}/${artifact.id}${format.suffix}',
      );
      if (format.isValid(modelDir)) return modelDir.path;
      final extractedPath = await _extractAppleModelZip(
        assetBytes,
        modelDir,
        format,
      );
      if (extractedPath != null) return extractedPath;
    }

    // A cached Core ML model with no bundled asset behind it was downloaded before the device had Core AI; drop it
    // so an iOS 27 upgrade does not keep both formats on disk.
    final staleCoreMLDir = Directory(
      '${directory.path}/${artifact.id}${_AppleModelFormat.coreML.suffix}',
    );
    if (preferred == _AppleModelFormat.coreAI && staleCoreMLDir.existsSync()) {
      staleCoreMLDir.deleteSync(recursive: true);
    }

    final archiveName = '${artifact.id}${preferred.archiveSuffix}';
    final archiveFile = File('${directory.path}/$archiveName');
    await _downloadToFile(
      '$_iosModelReleaseBaseUrl/$archiveName',
      archiveFile,
      progressId: artifact.id,
    );
    return _extractAppleModelArchiveFile(
      archiveFile,
      archiveName,
      preferredDir,
      preferred,
    );
  }

  static Future<String> _downloadRemoteModel(Uri uri) async {
    final documents = await getApplicationDocumentsDirectory();
    final fileName = uri.pathSegments.isEmpty ? 'model' : uri.pathSegments.last;
    final url = uri.toString();
    final isOfficialAsset =
        url.startsWith('$_androidModelReleaseBaseUrl/') ||
        url.startsWith('$_iosModelReleaseBaseUrl/');
    final directory = isOfficialAsset
        ? Directory('${documents.path}/$_officialModelCacheDirectory')
        : documents;

    final format = _AppleModelFormat.ofArchive(fileName);
    if (_isIosLikePlatform && format != null) {
      final modelName = fileName.replaceAll(format.archiveSuffix, '');
      final targetDir = Directory(
        '${directory.path}/$modelName${format.suffix}',
      );
      if (format.isValid(targetDir)) return targetDir.path;
      if (isOfficialAsset) {
        final legacyTargetDir = Directory(
          '${documents.path}/$modelName${format.suffix}',
        );
        if (legacyTargetDir.existsSync()) {
          legacyTargetDir.deleteSync(recursive: true);
        }
      }
      final archiveFile = File('${directory.path}/$fileName');
      await _downloadToFile(url, archiveFile, progressId: modelName);
      return _extractAppleModelArchiveFile(
        archiveFile,
        fileName,
        targetDir,
        format,
      );
    }

    final file = File('${directory.path}/$fileName');
    if (file.existsSync()) return file.path;
    if (isOfficialAsset) {
      final legacyFile = File('${documents.path}/$fileName');
      if (legacyFile.existsSync()) legacyFile.deleteSync();
    }
    await _downloadToFile(
      url,
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
    final fileName = assetPath.split('/').last;
    final format = _AppleModelFormat.ofArchive(fileName);
    if (format != null) {
      final modelName = fileName.replaceAll(format.archiveSuffix, '');
      final directory = await getApplicationDocumentsDirectory();
      final modelDir = Directory(
        '${directory.path}/$modelName${format.suffix}',
      );
      if (format.isValid(modelDir)) return modelDir.path;

      final assetBytes = await _loadAssetBytes(assetPath);
      if (assetBytes == null) {
        throw ModelLoadingException('Flutter asset not found: $assetPath');
      }

      final extractedPath = await _extractAppleModelZip(
        assetBytes,
        modelDir,
        format,
      );
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

  static Future<String> _extractAppleModelArchiveFile(
    File archiveFile,
    String displayName,
    Directory targetDir,
    _AppleModelFormat format,
  ) async {
    try {
      final extractedPath = await _extractAppleModelZip(
        archiveFile.readAsBytesSync(),
        targetDir,
        format,
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

  static Future<String?> _extractAppleModelZip(
    List<int> bytes,
    Directory targetDir,
    _AppleModelFormat format,
  ) async {
    try {
      if (targetDir.existsSync()) {
        targetDir.deleteSync(recursive: true);
      }
      targetDir.createSync(recursive: true);

      MiniZip.extractBytes(
        bytes,
        destination: targetDir,
        stripTopLevelDirectoryEndingWith: format.suffix,
        skip: (path) => path.startsWith('__MACOSX/') || path.contains('/._'),
      );

      return format.isValid(targetDir) ? targetDir.path : null;
    } catch (_) {
      if (targetDir.existsSync()) {
        targetDir.deleteSync(recursive: true);
      }
      return null;
    }
  }
}
