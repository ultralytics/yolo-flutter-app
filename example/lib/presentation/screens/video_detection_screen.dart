// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:video_player/video_player.dart';
import 'package:ultralytics_yolo/yolo.dart';

/// A screen that demonstrates YOLO inference on video files.
///
/// This screen allows users to:
/// - Pick a video from the gallery
/// - Process the video with YOLO detection
/// - View processing progress and results
class VideoDetectionScreen extends StatefulWidget {
  const VideoDetectionScreen({super.key});

  @override
  State<VideoDetectionScreen> createState() => _VideoDetectionScreenState();
}

class _VideoDetectionScreenState extends State<VideoDetectionScreen> {
  final _picker = ImagePicker();
  YOLO? _yolo;
  bool _isModelReady = false;
  bool _isProcessing = false;

  // Video processing state
  String? _selectedVideoPath;
  VideoPlayerController? _videoController;
  List<VideoFrameResult> _processingResults = [];
  double _processingProgress = 0.0;
  int _currentFrame = 0;
  int _totalFrames = 0;

  @override
  void initState() {
    super.initState();
    _initializeYOLO();
  }

  /// Initialize YOLO model for video processing
  Future<void> _initializeYOLO() async {
    try {
      // Use the model from assets
      _yolo = YOLO(
        modelPath: 'assets/models/yolo11n.mlpackage',
        task: YOLOTask.detect,
      );

      await _yolo!.loadModel();

      if (mounted) {
        setState(() {
          _isModelReady = true;
        });
      }

      debugPrint('YOLO model initialized for video processing');
    } catch (e) {
      debugPrint('Error loading YOLO model: $e');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error loading model: $e')));
      }
    }
  }

  /// Pick a video from the gallery
  Future<void> _pickVideo() async {
    try {
      final XFile? video = await _picker.pickVideo(
        source: ImageSource.gallery,
        maxDuration: const Duration(minutes: 5), // Limit to 5 minutes for demo
      );

      if (video != null) {
        setState(() {
          _selectedVideoPath = video.path;
        });

        // Initialize video player for preview
        await _initializeVideoPlayer(video.path);

        debugPrint('Video selected: ${video.path}');
      }
    } catch (e) {
      debugPrint('Error picking video: $e');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error picking video: $e')));
      }
    }
  }

  /// Initialize video player controller for preview
  Future<void> _initializeVideoPlayer(String videoPath) async {
    try {
      _videoController?.dispose();
      _videoController = VideoPlayerController.file(File(videoPath));
      await _videoController!.initialize();

      if (mounted) {
        setState(() {});
      }
    } catch (e) {
      debugPrint('Error initializing video player: $e');
    }
  }

  /// Process the selected video with YOLO detection
  Future<void> _processVideo() async {
    if (_yolo == null || _selectedVideoPath == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please select a video first')),
      );
      return;
    }

    setState(() {
      _isProcessing = true;
      _processingResults.clear();
      _processingProgress = 0.0;
      _currentFrame = 0;
      _totalFrames = 0;
    });

    try {
      final config = VideoProcessingConfig(
        frameRate: 1.0, // Process 1 frame per second
        confidenceThreshold: 0.25,
        iouThreshold: 0.4,
        saveProcessedFrames: false,
        onProgress: (progress, currentFrame, totalFrames) {
          if (mounted) {
            setState(() {
              _processingProgress = progress;
              _currentFrame = currentFrame;
              _totalFrames = totalFrames;
            });
          }
        },
        onComplete: (results, outputPath) {
          if (mounted) {
            setState(() {
              _processingResults = results;
              _isProcessing = false;
            });

            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(
                  'Video processing complete! Processed ${results.length} frames',
                ),
              ),
            );
          }
        },
        onError: (error) {
          if (mounted) {
            setState(() {
              _isProcessing = false;
            });

            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('Error processing video: $error')),
            );
          }
        },
      );

      await _yolo!.processVideo(videoPath: _selectedVideoPath!, config: config);
    } catch (e) {
      debugPrint('Error processing video: $e');
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });

        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error processing video: $e')));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Video Detection'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Video preview
            if (_videoController != null &&
                _videoController!.value.isInitialized)
              Container(
                height: 200,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: AspectRatio(
                    aspectRatio: _videoController!.value.aspectRatio,
                    child: VideoPlayer(_videoController!),
                  ),
                ),
              ),

            const SizedBox(height: 16),

            // Control buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isModelReady ? _pickVideo : null,
                    icon: const Icon(Icons.video_library),
                    label: const Text('Pick Video'),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _selectedVideoPath != null && !_isProcessing
                        ? _processVideo
                        : null,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text('Process Video'),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),

            // Processing progress
            if (_isProcessing) ...[
              LinearProgressIndicator(value: _processingProgress),
              const SizedBox(height: 8),
              Text(
                'Processing: $_currentFrame / $_totalFrames frames (${(_processingProgress * 100).toStringAsFixed(1)}%)',
                textAlign: TextAlign.center,
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 16),
            ],

            // Results
            if (_processingResults.isNotEmpty) ...[
              Text(
                'Processing Results (${_processingResults.length} frames)',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              Expanded(
                child: ListView.builder(
                  itemCount: _processingResults.length,
                  itemBuilder: (context, index) {
                    final result = _processingResults[index];
                    return Card(
                      child: ListTile(
                        title: Text('Frame ${result.frameIndex + 1}'),
                        subtitle: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Time: ${result.timestamp.toString().split('.').first}',
                            ),
                            if (result.error != null)
                              Text(
                                'Error: ${result.error}',
                                style: const TextStyle(color: Colors.red),
                              )
                            else if (result.detectionResult != null)
                              Text(
                                'Detections: ${result.detectionResult!.className} (${(result.detectionResult!.confidence * 100).toStringAsFixed(1)}%)',
                              )
                            else
                              const Text('No detections'),
                          ],
                        ),
                        leading: const Icon(Icons.videocam),
                      ),
                    );
                  },
                ),
              ),
            ],

            // Status message
            if (!_isModelReady)
              const Card(
                color: Colors.orange,
                child: Padding(
                  padding: EdgeInsets.all(16.0),
                  child: Text(
                    'Loading YOLO model...',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.white),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _videoController?.dispose();
    super.dispose();
  }
}
