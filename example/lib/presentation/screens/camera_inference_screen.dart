// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/yolo_result.dart';
import 'package:ultralytics_yolo/yolo_task.dart';
import 'package:ultralytics_yolo/yolo_view.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'package:archive/archive.dart';
import 'package:path/path.dart' as path;

class CameraInferenceScreen extends StatefulWidget {
  const CameraInferenceScreen({super.key});

  @override
  State<CameraInferenceScreen> createState() => _CameraInferenceScreenState();
}

enum SliderType { none, numItems, confidence, iou }

enum ModelType {
  detect('yolo11n', YOLOTask.detect),
  segment('yolo11n-seg', YOLOTask.segment),
  classify('yolo11n-cls', YOLOTask.classify),
  pose('yolo11n-pose', YOLOTask.pose),
  obb('yolo11n-obb', YOLOTask.obb),
  test('best', YOLOTask.detect); // Test model available in assets

  final String modelName;
  final YOLOTask task;
  const ModelType(this.modelName, this.task);
}

class _CameraInferenceScreenState extends State<CameraInferenceScreen> {
  int _detectionCount = 0;
  double _confidenceThreshold = 0.5;
  double _iouThreshold = 0.45;
  int _numItemsThreshold = 30;
  double _currentFps = 0.0;
  int _frameCount = 0;
  DateTime _lastFpsUpdate = DateTime.now();

  SliderType _activeSlider = SliderType.none;
  ModelType _selectedModel = ModelType.detect;
  bool _isModelLoading = false;
  String? _modelPath;
  String _loadingMessage = '';
  double _downloadProgress = 0.0;
  double _currentZoomLevel = 1.0;
  bool _isFrontCamera = false;

  final _yoloController = YoloViewController();
  final _yoloViewKey = GlobalKey<YoloViewState>();
  final bool _useController = true;

  void _onDetectionResults(List<YOLOResult> results) {
    if (!mounted) return;

    _frameCount++;
    final now = DateTime.now();
    final elapsed = now.difference(_lastFpsUpdate).inMilliseconds;

    if (elapsed >= 1000) {
      final calculatedFps = _frameCount * 1000 / elapsed;
      debugPrint('Calculated FPS: ${calculatedFps.toStringAsFixed(1)}');

      _currentFps = calculatedFps;
      _frameCount = 0;
      _lastFpsUpdate = now;
    }

    // Still update detection count in the UI
    setState(() {
      _detectionCount = results.length;
    });

    // Debug first few detections
    for (var i = 0; i < results.length && i < 3; i++) {
      final r = results[i];
      debugPrint(
        'Detection $i: ${r.className} (${(r.confidence * 100).toStringAsFixed(1)}%) at ${r.boundingBox}',
      );
    }
  }

  @override
  void initState() {
    super.initState();
    _loadModelForPlatform();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_useController) {
        _yoloController.setThresholds(
          confidenceThreshold: _confidenceThreshold,
          iouThreshold: _iouThreshold,
          numItemsThreshold: _numItemsThreshold,
        );
      } else {
        _yoloViewKey.currentState?.setThresholds(
          confidenceThreshold: _confidenceThreshold,
          iouThreshold: _iouThreshold,
          numItemsThreshold: _numItemsThreshold,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // YOLO View: must be at back
          if (_modelPath != null && !_isModelLoading)
            YoloView(
              key: _useController ? const ValueKey('yolo_view_static') : _yoloViewKey, // Use static key to prevent recreation
              controller: _useController ? _yoloController : null,
              modelPath: _modelPath!,
              task: _selectedModel.task,
              onResult: _onDetectionResults,
              onPerformanceMetrics: (metrics) {
                if (mounted) {
                  setState(() {
                    if (metrics['fps'] != null) {
                      _currentFps = metrics['fps']!;
                    }
                  });
                }
              },
              onZoomChanged: (zoomLevel) {
                if (mounted) {
                  setState(() {
                    _currentZoomLevel = zoomLevel;
                  });
                }
              },
            )
          else if (_isModelLoading)
            IgnorePointer(
              child: Container(
                color: Colors.black87,
                child: Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Ultralytics logo
                      Image.asset(
                        'assets/logo.png',
                        width: 120,
                        height: 120,
                        color: Colors.white.withOpacity(0.8),
                      ),
                      const SizedBox(height: 32),
                      // Loading message
                      Text(
                        _loadingMessage,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                        ),
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 24),
                      // Progress indicator
                      if (_downloadProgress > 0)
                        Column(
                          children: [
                            SizedBox(
                              width: 200,
                              child: LinearProgressIndicator(
                                value: _downloadProgress,
                                backgroundColor: Colors.white24,
                                valueColor: const AlwaysStoppedAnimation<Color>(
                                  Colors.white,
                                ),
                                minHeight: 4,
                              ),
                            ),
                            const SizedBox(height: 12),
                            Text(
                              '${(_downloadProgress * 100).toInt()}%',
                              style: const TextStyle(
                                color: Colors.white70,
                                fontSize: 14,
                              ),
                            ),
                          ],
                        )
                      else
                        const CircularProgressIndicator(color: Colors.white),
                    ],
                  ),
                ),
              ),
            ),

          // Top info pills (detection, FPS, and current threshold)
          Positioned(
            top: MediaQuery.of(context).padding.top + 16, // Safe area + spacing
            left: 16,
            right: 16,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                // Model selector
                _buildModelSelector(),
                const SizedBox(height: 12),
                IgnorePointer(
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(
                        'DETECTIONS: $_detectionCount',
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(width: 16),
                      Text(
                        'FPS: ${_currentFps.toStringAsFixed(1)}',
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 8),
                if (_activeSlider == SliderType.confidence)
                  _buildTopPill(
                    'CONFIDENCE THRESHOLD: ${_confidenceThreshold.toStringAsFixed(2)}',
                  ),
                if (_activeSlider == SliderType.iou)
                  _buildTopPill(
                    'IOU THRESHOLD: ${_iouThreshold.toStringAsFixed(2)}',
                  ),
                if (_activeSlider == SliderType.numItems)
                  _buildTopPill('ITEMS MAX: $_numItemsThreshold'),
              ],
            ),
          ),

          // Center logo - only show when camera is active
          if (_modelPath != null && !_isModelLoading)
            IgnorePointer(
              child: Positioned.fill(
                child: Align(
                  alignment: Alignment.center,
                  child: FractionallySizedBox(
                    widthFactor: 0.5,
                    heightFactor: 0.5,
                    child: Image.asset(
                      'assets/logo.png',
                      color: Colors.white.withOpacity(0.4),
                    ),
                  ),
                ),
              ),
            ),

          // Control buttons
          Positioned(
            bottom: 32,
            right: 16,
            child: Column(
              children: [
                if (!_isFrontCamera) ...[
                  _buildCircleButton(
                    '${_currentZoomLevel.toStringAsFixed(1)}x',
                    onPressed: () {
                      // Cycle through zoom levels: 0.5x -> 1.0x -> 3.0x -> 0.5x
                      double nextZoom;
                      if (_currentZoomLevel < 0.75) {
                        nextZoom = 1.0;
                      } else if (_currentZoomLevel < 2.0) {
                        nextZoom = 3.0;
                      } else {
                        nextZoom = 0.5;
                      }
                      _setZoomLevel(nextZoom);
                    },
                  ),
                  const SizedBox(height: 12),
                ],
                _buildIconButton(Icons.layers, () {
                  _toggleSlider(SliderType.numItems);
                }),
                const SizedBox(height: 12),
                _buildIconButton(Icons.adjust, () {
                  _toggleSlider(SliderType.confidence);
                }),
                const SizedBox(height: 12),
                _buildIconButton('assets/iou.png', () {
                  _toggleSlider(SliderType.iou);
                }),
                const SizedBox(height: 40),
              ],
            ),
          ),

          // Bottom slider overlay
          if (_activeSlider != SliderType.none)
            Positioned(
              left: 0,
              right: 0,
              bottom: 0,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 12,
                ),
                color: Colors.black.withOpacity(0.8),
                child: SliderTheme(
                  data: SliderTheme.of(context).copyWith(
                    activeTrackColor: Colors.yellow,
                    inactiveTrackColor: Colors.white.withOpacity(0.3),
                    thumbColor: Colors.yellow,
                    overlayColor: Colors.yellow.withOpacity(0.2),
                  ),
                  child: Slider(
                    value: _getSliderValue(),
                    min: _getSliderMin(),
                    max: _getSliderMax(),
                    divisions: _getSliderDivisions(),
                    label: _getSliderLabel(),
                    onChanged: (value) {
                      setState(() {
                        _updateSliderValue(value);
                      });
                    },
                  ),
                ),
              ),
            ),
          // Camera flip top-right
          Positioned(
            bottom: MediaQuery.of(context).padding.top + 16,
            left: 16,
            child: CircleAvatar(
              radius: 24,
              backgroundColor: Colors.black.withOpacity(0.5),
              child: IconButton(
                icon: const Icon(Icons.flip_camera_ios, color: Colors.white),
                onPressed: () {
                  setState(() {
                    _isFrontCamera = !_isFrontCamera;
                    // Reset zoom level when switching to front camera
                    if (_isFrontCamera) {
                      _currentZoomLevel = 1.0;
                    }
                  });
                  if (_useController) {
                    _yoloController.switchCamera();
                  } else {
                    _yoloViewKey.currentState?.switchCamera();
                  }
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildIconButton(dynamic iconOrAsset, VoidCallback onPressed) {
    return CircleAvatar(
      radius: 24,
      backgroundColor: Colors.black.withOpacity(0.2),
      child: IconButton(
        icon: iconOrAsset is IconData
            ? Icon(iconOrAsset, color: Colors.white)
            : Image.asset(
                iconOrAsset,
                width: 24,
                height: 24,
                color: Colors.white,
              ),
        onPressed: onPressed,
      ),
    );
  }

  // Zoom circle button
  Widget _buildCircleButton(String label, {required VoidCallback onPressed}) {
    return CircleAvatar(
      radius: 24,
      backgroundColor: Colors.black.withOpacity(0.2),
      child: TextButton(
        onPressed: onPressed,
        child: Text(label, style: const TextStyle(color: Colors.white)),
      ),
    );
  }

  // Toggle slider state
  void _toggleSlider(SliderType type) {
    setState(() {
      _activeSlider = (_activeSlider == type) ? SliderType.none : type;
    });
  }

  Widget _buildTopPill(String label) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.6),
        borderRadius: BorderRadius.circular(24),
      ),
      child: Text(
        label,
        style: const TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  // Slider value helpers
  double _getSliderValue() {
    switch (_activeSlider) {
      case SliderType.numItems:
        return _numItemsThreshold.toDouble();
      case SliderType.confidence:
        return _confidenceThreshold;
      case SliderType.iou:
        return _iouThreshold;
      default:
        return 0;
    }
  }

  double _getSliderMin() => _activeSlider == SliderType.numItems ? 5 : 0.1;

  double _getSliderMax() => _activeSlider == SliderType.numItems ? 50 : 0.9;

  int _getSliderDivisions() => _activeSlider == SliderType.numItems ? 9 : 8;

  String _getSliderLabel() {
    switch (_activeSlider) {
      case SliderType.numItems:
        return '$_numItemsThreshold';
      case SliderType.confidence:
        return _confidenceThreshold.toStringAsFixed(1);
      case SliderType.iou:
        return _iouThreshold.toStringAsFixed(1);
      default:
        return '';
    }
  }

  // 🧠 Slider value update logic
  void _updateSliderValue(double value) {
    switch (_activeSlider) {
      case SliderType.numItems:
        _numItemsThreshold = value.toInt();
        if (_useController) {
          _yoloController.setNumItemsThreshold(_numItemsThreshold);
        } else {
          _yoloViewKey.currentState?.setNumItemsThreshold(_numItemsThreshold);
        }
        break;
      case SliderType.confidence:
        _confidenceThreshold = value;
        if (_useController) {
          _yoloController.setConfidenceThreshold(value);
        } else {
          _yoloViewKey.currentState?.setConfidenceThreshold(value);
        }
        break;
      case SliderType.iou:
        _iouThreshold = value;
        if (_useController) {
          _yoloController.setIoUThreshold(value);
        } else {
          _yoloViewKey.currentState?.setIoUThreshold(value);
        }
        break;
      default:
        break;
    }
  }

  void _setZoomLevel(double zoomLevel) {
    setState(() {
      _currentZoomLevel = zoomLevel;
    });
    if (_useController) {
      _yoloController.setZoomLevel(zoomLevel);
    } else {
      _yoloViewKey.currentState?.setZoomLevel(zoomLevel);
    }
  }

  Widget _buildModelSelector() {
    return Container(
      height: 36,
      padding: const EdgeInsets.all(2),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.6),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: ModelType.values.map((model) {
          final isSelected = _selectedModel == model;
          return GestureDetector(
            onTap: () {
              if (!_isModelLoading && model != _selectedModel) {
                setState(() {
                  _selectedModel = model;
                });
                _loadModelForPlatform();
              }
            },
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
              decoration: BoxDecoration(
                color: isSelected ? Colors.white : Colors.transparent,
                borderRadius: BorderRadius.circular(6),
              ),
              child: Text(
                model.name.toUpperCase(),
                style: TextStyle(
                  color: isSelected ? Colors.black : Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  Future<void> _loadModelForPlatform() async {
    setState(() {
      _isModelLoading = true;
      _loadingMessage = 'Loading ${_selectedModel.modelName} model...';
      _downloadProgress = 0.0;
      // Reset metrics when switching models
      _detectionCount = 0;
      _currentFps = 0.0;
      _frameCount = 0;
      _lastFpsUpdate = DateTime.now();
    });

    try {
      String? modelPath;

      if (Platform.isIOS) {
        // Try local bundle first
        // If not found, download and extract mlpackage.zip
        modelPath = await _getIOSModelPath();
      } else if (Platform.isAndroid) {
        // Try local bundle first (model name without extension)
        // If not found, download tflite
        modelPath = await _getAndroidModelPath();
      }

      if (mounted) {
        setState(() {
          _modelPath = modelPath;
          _isModelLoading = false;
          _loadingMessage = '';
          _downloadProgress = 0.0;
        });

        // If modelPath is null, show error
        if (modelPath == null) {
          showDialog(
            context: context,
            builder: (context) => AlertDialog(
              title: const Text('Model Not Found'),
              content: Text(
                Platform.isIOS
                    ? 'The ${_selectedModel.modelName} model is not available for iOS. iOS models need to be bundled with the app or downloaded separately.\n\nFor testing, you can use the TEST model on Android.'
                    : 'The ${_selectedModel.modelName} model is not bundled. Try using the TEST model which is available in assets.',
              ),
              actions: [
                TextButton(
                  onPressed: () {
                    Navigator.pop(context);
                    if (Platform.isAndroid &&
                        _selectedModel != ModelType.test) {
                      setState(() {
                        _selectedModel = ModelType.test;
                      });
                      _loadModelForPlatform();
                    }
                  },
                  child: Text(Platform.isAndroid ? 'Use Test Model' : 'OK'),
                ),
              ],
            ),
          );
        }
      }
    } catch (e) {
      debugPrint('Error loading model: $e');
      if (mounted) {
        setState(() {
          _isModelLoading = false;
          _loadingMessage = 'Failed to load model';
          _downloadProgress = 0.0;
        });
        // Show error dialog
        showDialog(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Model Loading Error'),
            content: Text(
              'Failed to load ${_selectedModel.modelName} model. Please try again.',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('OK'),
              ),
            ],
          ),
        );
      }
    }
  }

  Future<String?> _getIOSModelPath() async {
    // Update message for checking
    if (mounted) {
      setState(() {
        _loadingMessage = 'Loading ${_selectedModel.modelName} model...';
      });
    }

    // Update message for checking
    if (mounted) {
      setState(() {
        _loadingMessage = 'Checking for ${_selectedModel.modelName} model...';
      });
    }

    // For iOS, we need to return the full path to the mlpackage directory
    // Models are stored in the documents directory after download/extraction
    final documentsDir = await getApplicationDocumentsDirectory();
    final modelDir = Directory(
      '${documentsDir.path}/${_selectedModel.modelName}.mlpackage',
    );

    if (await modelDir.exists()) {
      debugPrint('Found existing iOS model at: ${modelDir.path}');
      return modelDir.path;
    }

    // Update message for downloading
    if (mounted) {
      setState(() {
        _loadingMessage = 'Downloading ${_selectedModel.modelName} model...';
      });
    }

    // Download model from GitHub
    final url =
        'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.0.0/${_selectedModel.modelName}.mlpackage.zip';

    try {
      final request = await http.Client().send(
        http.Request('GET', Uri.parse(url)),
      );
      final contentLength = request.contentLength ?? 0;

      // Download with progress tracking
      final bytes = <int>[];
      int downloadedBytes = 0;

      await for (final chunk in request.stream) {
        bytes.addAll(chunk);
        downloadedBytes += chunk.length;

        if (contentLength > 0 && mounted) {
          setState(() {
            _downloadProgress = downloadedBytes / contentLength;
          });
        }
      }

      if (request.statusCode == 200) {
        if (mounted) {
          setState(() {
            _loadingMessage = 'Extracting ${_selectedModel.modelName} model...';
            _downloadProgress = 1.0;
          });
        }

        // Save zip file temporarily
        final zipFile = File(
          '${documentsDir.path}/${_selectedModel.modelName}.mlpackage.zip',
        );
        await zipFile.writeAsBytes(bytes);

        // Extract zip
        try {
          final archive = ZipDecoder().decodeBytes(bytes);

          // Create the mlpackage directory
          await modelDir.create(recursive: true);

          // Extract files with prefix handling
          for (final file in archive) {
            if (file.isFile) {
              // Handle various zip structure patterns
              String targetPath = file.name;

              // Remove common prefixes that might exist in the zip
              final prefixes = [
                '${_selectedModel.modelName}.mlpackage/',
                '${_selectedModel.modelName}/',
                'mlpackage/',
              ];

              for (final prefix in prefixes) {
                if (targetPath.startsWith(prefix)) {
                  targetPath = targetPath.substring(prefix.length);
                  break;
                }
              }

              // Create the full path within the mlpackage directory
              final fullPath = path.join(modelDir.path, targetPath);
              final outFile = File(fullPath);

              // Create parent directories if needed
              await outFile.parent.create(recursive: true);
              await outFile.writeAsBytes(file.content as List<int>);
            }
          }

          // Delete the zip file after extraction
          await zipFile.delete();

          // Verify the mlpackage directory exists
          if (await modelDir.exists()) {
            return modelDir.path;
          } else {
            debugPrint('Error: mlpackage directory not found after extraction');
          }
        } catch (e) {
          debugPrint('Failed to extract mlpackage: $e');
          if (await zipFile.exists()) {
            await zipFile.delete();
          }
          // Clean up the model directory if extraction failed
          if (await modelDir.exists()) {
            await modelDir.delete(recursive: true);
          }
        }
      }
    } catch (e) {
      debugPrint('Failed to download iOS model: $e');
    }

    // Return null if download/extraction failed
    return null;
  }

  Future<String?> _getAndroidModelPath() async {
    // Update message for checking
    if (mounted) {
      setState(() {
        _loadingMessage = 'Checking for ${_selectedModel.modelName} model...';
      });
    }

    // First check if model exists in assets (bundled)
    final bundledModelName = '${_selectedModel.modelName}.tflite';

    // Only test model is bundled
    if (_selectedModel == ModelType.test) {
      // Test model is known to be bundled
      debugPrint('Using bundled Android model: $bundledModelName');
      return bundledModelName;
    }

    // For all other models (including detect), try to download them
    final documentsDir = await getApplicationDocumentsDirectory();
    final modelFile = File(
      '${documentsDir.path}/${_selectedModel.modelName}.tflite',
    );

    if (await modelFile.exists()) {
      debugPrint('Found existing Android model at: ${modelFile.path}');
      return modelFile.path;
    }

    // Update message for downloading
    if (mounted) {
      setState(() {
        _loadingMessage = 'Downloading ${_selectedModel.modelName} model...';
      });
    }

    // Download model from GitHub
    final url =
        'https://github.com/ultralytics/yolo-flutter-app/releases/download/v0.0.0/${_selectedModel.modelName}.tflite';

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

        if (contentLength > 0 && mounted) {
          setState(() {
            _downloadProgress = downloadedBytes / contentLength;
          });
        }
      }

      if (request.statusCode == 200) {
        if (mounted) {
          setState(() {
            _loadingMessage = 'Saving ${_selectedModel.modelName} model...';
          });
        }

        await modelFile.writeAsBytes(bytes);
        return modelFile.path;
      }
    } catch (e) {
      debugPrint('Failed to download Android model: $e');
    }

    // If download failed, try bundled model as fallback
    return bundledModelName;
  }
}
