// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'package:video_thumbnail/video_thumbnail.dart';
import 'package:path_provider/path_provider.dart';
import 'package:ultralytics_yolo/yolo.dart';

/// Callback for video processing progress updates
typedef VideoProcessingProgressCallback =
    void Function(double progress, int currentFrame, int totalFrames);

/// Callback for video processing completion
typedef VideoProcessingCompleteCallback =
    void Function(List<VideoFrameResult> results, String? outputPath);

/// Callback for video processing error
typedef VideoProcessingErrorCallback = void Function(String error);

/// Result of processing a single video frame
class VideoFrameResult {
  final int frameIndex;
  final Duration timestamp;
  final Uint8List? frameData;
  final YOLOResult? detectionResult;
  final String? error;

  VideoFrameResult({
    required this.frameIndex,
    required this.timestamp,
    this.frameData,
    this.detectionResult,
    this.error,
  });

  Map<String, dynamic> toMap() {
    return {
      'frameIndex': frameIndex,
      'timestamp': timestamp.inMilliseconds,
      'frameData': frameData,
      'detectionResult': detectionResult?.toMap(),
      'error': error,
    };
  }

  factory VideoFrameResult.fromMap(Map<String, dynamic> map) {
    return VideoFrameResult(
      frameIndex: map['frameIndex'] as int,
      timestamp: Duration(milliseconds: map['timestamp'] as int),
      frameData: map['frameData'] as Uint8List?,
      detectionResult: map['detectionResult'] != null
          ? YOLOResult.fromMap(
              Map<String, dynamic>.from(map['detectionResult']),
            )
          : null,
      error: map['error'] as String?,
    );
  }
}

/// Configuration for video processing
class VideoProcessingConfig {
  final double frameRate;
  final double confidenceThreshold;
  final double iouThreshold;
  final bool saveProcessedFrames;
  final String? outputDirectory;
  final VideoProcessingProgressCallback? onProgress;
  final VideoProcessingCompleteCallback? onComplete;
  final VideoProcessingErrorCallback? onError;

  const VideoProcessingConfig({
    this.frameRate = 1.0,
    this.confidenceThreshold = 0.25,
    this.iouThreshold = 0.4,
    this.saveProcessedFrames = false,
    this.outputDirectory,
    this.onProgress,
    this.onComplete,
    this.onError,
  });
}

/// Handles video processing for YOLO detection
class VideoProcessor {
  static const String _tag = 'VideoProcessor';

  VideoPlayerController? _controller;
  YOLO? _yolo;
  bool _isProcessing = false;
  StreamSubscription? _processingSubscription;

  /// Initialize video processor with YOLO model
  Future<void> initialize({
    required YOLO yolo,
    required String videoPath,
  }) async {
    try {
      _yolo = yolo;

      // Initialize video player controller
      _controller = VideoPlayerController.file(File(videoPath));
      await _controller!.initialize();

      debugPrint(
        '$_tag: Video initialized - Duration: ${_controller!.value.duration}',
      );
    } catch (e) {
      debugPrint('$_tag: Error initializing video processor: $e');
      rethrow;
    }
  }

  /// Get video information
  VideoInfo getVideoInfo() {
    if (_controller == null) {
      throw StateError('Video processor not initialized');
    }

    return VideoInfo(
      duration: _controller!.value.duration,
      size: _controller!.value.size,
      frameRate:
          30.0, // Default frame rate, can be estimated from video metadata
    );
  }

  /// Process video with YOLO detection
  Future<List<VideoFrameResult>> processVideo({
    required VideoProcessingConfig config,
  }) async {
    if (_controller == null || _yolo == null) {
      throw StateError('Video processor not initialized');
    }

    if (_isProcessing) {
      throw StateError('Video processing already in progress');
    }

    _isProcessing = true;
    final results = <VideoFrameResult>[];

    try {
      final videoInfo = getVideoInfo();
      final totalFrames =
          (videoInfo.duration.inMilliseconds / 1000 * config.frameRate).round();

      debugPrint(
        '$_tag: Starting video processing - Total frames: $totalFrames',
      );

      // Calculate frame interval in milliseconds
      final frameInterval = (1000 / config.frameRate).round();

      for (int frameIndex = 0; frameIndex < totalFrames; frameIndex++) {
        if (!_isProcessing) break; // Allow cancellation

        final timestamp = Duration(milliseconds: frameIndex * frameInterval);

        try {
          // Extract frame at specific timestamp
          final frameData = await _extractFrameAtTimestamp(timestamp);

          if (frameData != null) {
            // Run YOLO detection on the frame
            final detectionResult = await _yolo!.predict(
              frameData,
              confidenceThreshold: config.confidenceThreshold,
              iouThreshold: config.iouThreshold,
            );

            final result = VideoFrameResult(
              frameIndex: frameIndex,
              timestamp: timestamp,
              frameData: config.saveProcessedFrames ? frameData : null,
              detectionResult: YOLOResult.fromMap(detectionResult),
            );

            results.add(result);

            // Report progress
            final progress = (frameIndex + 1) / totalFrames;
            config.onProgress?.call(progress, frameIndex + 1, totalFrames);

            debugPrint(
              '$_tag: Processed frame $frameIndex - Detections: ${result.detectionResult != null ? 1 : 0}',
            );
          }
        } catch (e) {
          debugPrint('$_tag: Error processing frame $frameIndex: $e');

          results.add(
            VideoFrameResult(
              frameIndex: frameIndex,
              timestamp: timestamp,
              error: e.toString(),
            ),
          );
        }
      }

      config.onComplete?.call(results, null);
      return results;
    } catch (e) {
      final error = 'Error processing video: $e';
      debugPrint('$_tag: $error');
      config.onError?.call(error);
      rethrow;
    } finally {
      _isProcessing = false;
    }
  }

  /// Extract a single frame at the specified timestamp
  Future<Uint8List?> _extractFrameAtTimestamp(Duration timestamp) async {
    try {
      await _controller!.seekTo(timestamp);

      await Future.delayed(const Duration(milliseconds: 100));

      // Extract frame using video_thumbnail
      final tempDir = await getTemporaryDirectory();
      final thumbnailPath =
          '${tempDir.path}/frame_${timestamp.inMilliseconds}.jpg';

      final thumbnail = await VideoThumbnail.thumbnailFile(
        video: _controller!.dataSource,
        thumbnailPath: thumbnailPath,
        imageFormat: ImageFormat.JPEG,
        timeMs: timestamp.inMilliseconds,
        quality: 80,
      );

      if (thumbnail != null) {
        final file = File(thumbnail);
        final bytes = await file.readAsBytes();

        await file.delete(); // temporary file

        return bytes;
      }

      return null;
    } catch (e) {
      debugPrint(
        '$_tag: Error extracting frame at ${timestamp.inMilliseconds}ms: $e',
      );
      return null;
    }
  }

  /// Cancel ongoing video processing
  void cancelProcessing() {
    _isProcessing = false;
    _processingSubscription?.cancel();
    debugPrint('$_tag: Video processing cancelled');
  }

  /// Dispose resources
  Future<void> dispose() async {
    cancelProcessing();
    await _controller?.dispose();
    _controller = null;
    _yolo = null;
  }
}

/// Video information
class VideoInfo {
  final Duration duration;
  final Size size;
  final double frameRate;

  VideoInfo({
    required this.duration,
    required this.size,
    required this.frameRate,
  });

  @override
  String toString() {
    return 'VideoInfo(duration: $duration, size: $size, frameRate: $frameRate)';
  }
}
