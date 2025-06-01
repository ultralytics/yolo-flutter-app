// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'package:flutter/material.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'models/model_type.dart';
import 'services/model_manager.dart';

/// Simple streaming test screen with minimal/full toggle and real-time data indicators
class StreamingTestScreen extends StatefulWidget {
  const StreamingTestScreen({super.key});

  @override
  State<StreamingTestScreen> createState() => _StreamingTestScreenState();
}

class _StreamingTestScreenState extends State<StreamingTestScreen> {
  final _yoloController = YOLOViewController();
  
  // Streaming configuration
  bool _isFullMode = false; // false = minimal, true = full
  int? _inferenceFrequency; // null = max frequency
  bool _showInferenceControls = false;
  
  // Task selection
  YOLOTask _selectedTask = YOLOTask.segment;
  
  // Real-time streaming data status
  double _currentFps = 0.0;
  double _processingTimeMs = 0.0;
  int _detectionCount = 0;
  
  // Data availability indicators
  bool _hasDetections = false;
  bool _hasMasks = false;
  bool _hasPoses = false;
  bool _hasOBB = false;
  bool _hasOriginalImage = false;
  
  // Model configuration
  String? _modelPath;
  bool _isModelLoading = true;
  late final ModelManager _modelManager;

  @override
  void initState() {
    super.initState();
    
    debugPrint('StreamingTest: Initializing app...');
    
    // Initialize ModelManager
    _modelManager = ModelManager(
      onDownloadProgress: (progress) {
        debugPrint('ModelManager: Download progress: $progress');
      },
      onStatusUpdate: (message) {
        debugPrint('ModelManager: $message');
      },
    );
    
    _loadDefaultModel();
    
    // Set initial streaming configuration after frame
    WidgetsBinding.instance.addPostFrameCallback((_) {
      debugPrint('StreamingTest: Post frame callback - calling _updateStreamingConfig');
      _updateStreamingConfig();
    });
  }

  Future<void> _loadDefaultModel() async {
    setState(() {
      _isModelLoading = true;
    });
    
    try {
      debugPrint('StreamingTest: Starting model loading...');
      // Use ModelManager to get model for selected task
      final modelType = _getModelTypeForTask(_selectedTask);
      final modelPath = await _modelManager.getModelPath(modelType);
      
      debugPrint('StreamingTest: Model loaded at path: $modelPath');
      setState(() {
        _modelPath = modelPath;
        _isModelLoading = false;
      });
      
      // Wait a bit then try to update streaming config
      Future.delayed(const Duration(milliseconds: 500), () {
        _updateStreamingConfig();
      });
    } catch (e) {
      debugPrint('StreamingTest: Error loading model: $e');
      setState(() {
        _isModelLoading = false;
      });
    }
  }

  void _updateStreamingConfig() {
    if (_yoloController.isInitialized) {
      final config = _isFullMode 
          ? YOLOStreamingConfig(
              includeDetections: true,
              includeClassifications: true,
              includeProcessingTimeMs: true,
              includeFps: true,
              includeMasks: true,
              includePoses: true,
              includeOBB: true,
              includeOriginalImage: false,
              inferenceFrequency: _inferenceFrequency,
            )
          : YOLOStreamingConfig(
              includeDetections: true,
              includeClassifications: true,
              includeProcessingTimeMs: true,
              includeFps: true,
              includeMasks: false,
              includePoses: false,
              includeOBB: false,
              includeOriginalImage: false,
              inferenceFrequency: _inferenceFrequency,
            );
      
      debugPrint('StreamingTest: 🔄 Updating streaming config');
      debugPrint('StreamingTest: Mode: ${_isFullMode ? "FULL" : "MINIMAL"}');
      debugPrint('StreamingTest: Include Masks: ${config.includeMasks}');
      debugPrint('StreamingTest: Include Poses: ${config.includePoses}');
      debugPrint('StreamingTest: Include OBB: ${config.includeOBB}');
      debugPrint('StreamingTest: Inference Frequency: ${_inferenceFrequency ?? "AUTO"}');
      debugPrint('StreamingTest: Task: $_selectedTask');
      
      _yoloController.setStreamingConfig(config);
      debugPrint('StreamingTest: ✅ Streaming config sent to controller');
    } else {
      debugPrint('StreamingTest: ⚠️ Controller not initialized, retrying...');
      // Retry after a short delay if not initialized
      Future.delayed(const Duration(milliseconds: 100), () {
        if (mounted) _updateStreamingConfig();
      });
    }
  }

  void _onStreamingData(Map<String, dynamic> streamData) {
    if (!mounted) return;
    
    debugPrint('StreamingTest: 📊 Received streaming data: ${streamData.keys.toList()}');
    
    setState(() {
      
      // Extract performance metrics
      _currentFps = (streamData['fps'] as num?)?.toDouble() ?? 0.0;
      _processingTimeMs = (streamData['processingTimeMs'] as num?)?.toDouble() ?? 0.0;
      
      debugPrint('StreamingTest: 📈 FPS: $_currentFps, Processing: $_processingTimeMs ms');
      
      // Check data availability
      final detections = streamData['detections'] as List?;
      _detectionCount = detections?.length ?? 0;
      _hasDetections = detections != null && detections.isNotEmpty;
      
      debugPrint('StreamingTest: 🔍 Detections count: $_detectionCount');
      
      // Check for different data types in first detection
      if (_hasDetections && detections!.isNotEmpty) {
        final firstDetection = detections.first as Map<String, dynamic>;
        _hasMasks = firstDetection.containsKey('mask');
        _hasPoses = firstDetection.containsKey('keypoints');
        _hasOBB = firstDetection.containsKey('obb');
        
        debugPrint('StreamingTest: 🎭 Mask data available: $_hasMasks');
        debugPrint('StreamingTest: 🤸 Pose data available: $_hasPoses');
        debugPrint('StreamingTest: 📦 OBB data available: $_hasOBB');
        
        if (_hasMasks) {
          final maskData = firstDetection['mask'];
          debugPrint('StreamingTest: ✅ Mask data type: ${maskData.runtimeType}');
          if (maskData is List && maskData.isNotEmpty) {
            debugPrint('StreamingTest: 📐 Mask dimensions: ${maskData.length}x${maskData.first?.length ?? 0}');
          }
        }
      } else {
        _hasMasks = false;
        _hasPoses = false;
        _hasOBB = false;
      }
      
      _hasOriginalImage = streamData.containsKey('originalImage');
      debugPrint('StreamingTest: 🖼️ Original image available: $_hasOriginalImage');
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // YOLO Camera View
          if (_modelPath != null && !_isModelLoading)
            YOLOView(
              controller: _yoloController,
              modelPath: _modelPath!,
              task: _selectedTask,
              // Use separated callbacks (confirmed working)
              onResult: (results) {
                debugPrint('StreamingTest: 📋 onResult called with ${results.length} results');
                
                // Check for mask data in results
                bool foundMasks = false;
                bool foundPoses = false;
                bool foundOBB = false;
                
                for (final result in results) {
                  if (result.mask != null && result.mask!.isNotEmpty) {
                    foundMasks = true;
                    debugPrint('StreamingTest: ✅ Found mask data in result for ${result.className}: ${result.mask!.length}x${result.mask!.first.length}');
                  }
                  if (result.keypoints != null && result.keypoints!.isNotEmpty) {
                    foundPoses = true;
                    debugPrint('StreamingTest: ✅ Found pose data in result for ${result.className}: ${result.keypoints!.length} keypoints');
                  }
                  // Note: OBB data is not currently available in YOLOResult separated callback mode
                }
                
                setState(() {
                  _detectionCount = results.length;
                  _hasDetections = results.isNotEmpty;
                  _hasMasks = foundMasks;
                  _hasPoses = foundPoses;
                  // _hasOBB remains false for separated callback mode
                });
              },
              onPerformanceMetrics: (metrics) {
                debugPrint('StreamingTest: 📊 onPerformanceMetrics called - FPS: ${metrics.fps}, Time: ${metrics.processingTimeMs}ms');
                setState(() {
                  _currentFps = metrics.fps;
                  _processingTimeMs = metrics.processingTimeMs;
                });
              },
              streamingConfig: _isFullMode 
                  ? YOLOStreamingConfig(
                      includeDetections: true,
                      includeClassifications: true,
                      includeProcessingTimeMs: true,
                      includeFps: true,
                      includeMasks: true,
                      includePoses: true,
                      includeOBB: true,
                      includeOriginalImage: false,
                      inferenceFrequency: _inferenceFrequency,
                    )
                  : YOLOStreamingConfig(
                      includeDetections: true,
                      includeClassifications: true,
                      includeProcessingTimeMs: true,
                      includeFps: true,
                      includeMasks: false,
                      includePoses: false,
                      includeOBB: false,
                      includeOriginalImage: false,
                      inferenceFrequency: _inferenceFrequency,
                    ),
            ),
          
          // Loading overlay
          if (_isModelLoading)
            Container(
              color: Colors.black87,
              child: const Center(
                child: CircularProgressIndicator(color: Colors.white),
              ),
            ),
          
          // Top Status Bar
          Positioned(
            top: MediaQuery.of(context).padding.top + 16,
            left: 16,
            right: 16,
            child: _buildTopStatusBar(),
          ),
          
          // Bottom Control Panel
          Positioned(
            bottom: MediaQuery.of(context).padding.bottom + 16,
            left: 16,
            right: 16,
            child: _buildBottomControlPanel(),
          ),
          
          // Task Selection (Top Right)
          Positioned(
            top: MediaQuery.of(context).padding.top + 16,
            right: 16,
            child: _buildTaskSelector(),
          ),
        ],
      ),
    );
  }

  Widget _buildTopStatusBar() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.8),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          // Performance metrics
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _buildMetricTile('FPS', _currentFps.toStringAsFixed(1), Colors.green),
              _buildMetricTile('MS', _processingTimeMs.toStringAsFixed(3), Colors.blue),
              _buildMetricTile('DETECTIONS', _detectionCount.toString(), Colors.orange),
            ],
          ),
          
          const SizedBox(height: 12),
          
          // Data availability indicators
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _buildDataIndicator('DETECT', _hasDetections),
              _buildDataIndicator('MASKS', _hasMasks),
              _buildDataIndicator('POSES', _hasPoses),
              _buildDataIndicator('OBB', _hasOBB),
              _buildDataIndicator('IMAGE', _hasOriginalImage),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildBottomControlPanel() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.8),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Mode toggle
          Row(
            children: [
              const Text(
                'Streaming Mode:',
                style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
              ),
              const Spacer(),
              Switch(
                value: _isFullMode,
                onChanged: (value) {
                  debugPrint('StreamingTest: 🔄 User toggled mode from ${_isFullMode ? "FULL" : "MINIMAL"} to ${value ? "FULL" : "MINIMAL"}');
                  setState(() {
                    _isFullMode = value;
                  });
                  debugPrint('StreamingTest: 📝 State updated - _isFullMode: $_isFullMode');
                  _updateStreamingConfig();
                },
                activeColor: Colors.orange,
                inactiveThumbColor: Colors.green,
                inactiveTrackColor: Colors.green.withOpacity(0.3),
                activeTrackColor: Colors.orange.withOpacity(0.3),
              ),
              Text(
                _isFullMode ? 'FULL' : 'MINIMAL',
                style: TextStyle(
                  color: _isFullMode ? Colors.orange : Colors.green,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          
          const SizedBox(height: 8),
          
          // Mode description
          Text(
            _isFullMode 
                ? 'Full mode: All data types (slower, more data)'
                : 'Minimal mode: Basic detection only (faster)',
            style: TextStyle(
              color: Colors.white.withOpacity(0.7),
              fontSize: 12,
            ),
            textAlign: TextAlign.center,
          ),
          
          const SizedBox(height: 16),
          
          // Inference frequency controls
          Row(
            children: [
              const Text(
                'Inference Frequency:',
                style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
              ),
              const Spacer(),
              GestureDetector(
                onTap: () {
                  setState(() {
                    _showInferenceControls = !_showInferenceControls;
                  });
                },
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.purple.withOpacity(0.2),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: Colors.purple.withOpacity(0.5)),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        _inferenceFrequency == null ? 'AUTO' : '${_inferenceFrequency}FPS',
                        style: const TextStyle(
                          color: Colors.purple,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(width: 4),
                      Icon(
                        _showInferenceControls ? Icons.expand_less : Icons.expand_more,
                        color: Colors.purple,
                        size: 16,
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
          
          // Inference frequency options
          if (_showInferenceControls) ...[
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _buildInferenceFrequencyButton(null, 'AUTO'),
                _buildInferenceFrequencyButton(30, '30'),
                _buildInferenceFrequencyButton(20, '20'),
                _buildInferenceFrequencyButton(15, '15'),
                _buildInferenceFrequencyButton(10, '10'),
                _buildInferenceFrequencyButton(5, '5'),
              ],
            ),
            const SizedBox(height: 8),
            Text(
              _inferenceFrequency == null
                  ? 'Auto: Maximum inference frequency'
                  : 'Fixed: $_inferenceFrequency inferences per second',
              style: TextStyle(
                color: Colors.white.withOpacity(0.7),
                fontSize: 12,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildMetricTile(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          label,
          style: TextStyle(
            color: Colors.white.withOpacity(0.7),
            fontSize: 12,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 4),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color: color.withOpacity(0.2),
            borderRadius: BorderRadius.circular(6),
            border: Border.all(color: color.withOpacity(0.5)),
          ),
          child: Text(
            value,
            style: TextStyle(
              color: color,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildDataIndicator(String label, bool hasData) {
    final color = hasData ? Colors.green : Colors.red;
    
    return Column(
      children: [
        Text(
          label,
          style: TextStyle(
            color: Colors.white.withOpacity(0.7),
            fontSize: 10,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 4),
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
            boxShadow: hasData ? [
              BoxShadow(
                color: color.withOpacity(0.5),
                blurRadius: 4,
                spreadRadius: 1,
              ),
            ] : null,
          ),
        ),
      ],
    );
  }

  Widget _buildInferenceFrequencyButton(int? frequency, String label) {
    final isSelected = _inferenceFrequency == frequency;
    
    return GestureDetector(
      onTap: () {
        setState(() {
          _inferenceFrequency = frequency;
        });
        _updateStreamingConfig();
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected 
              ? Colors.purple.withOpacity(0.3) 
              : Colors.white.withOpacity(0.1),
          borderRadius: BorderRadius.circular(6),
          border: Border.all(
            color: isSelected 
                ? Colors.purple 
                : Colors.white.withOpacity(0.3),
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            color: isSelected ? Colors.purple : Colors.white,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            fontSize: 12,
          ),
        ),
      ),
    );
  }

  ModelType _getModelTypeForTask(YOLOTask task) {
    switch (task) {
      case YOLOTask.detect:
        return ModelType.detect;
      case YOLOTask.segment:
        return ModelType.segment;
      case YOLOTask.classify:
        return ModelType.classify;
      case YOLOTask.pose:
        return ModelType.pose;
      case YOLOTask.obb:
        return ModelType.obb;
    }
  }

  void _changeTask(YOLOTask newTask) async {
    if (newTask == _selectedTask) return;
    
    setState(() {
      _selectedTask = newTask;
      _isModelLoading = true;
    });
    
    try {
      debugPrint('StreamingTest: Changing task to: $newTask');
      final modelType = _getModelTypeForTask(newTask);
      final modelPath = await _modelManager.getModelPath(modelType);
      
      setState(() {
        _modelPath = modelPath;
        _isModelLoading = false;
      });
      
      debugPrint('StreamingTest: Task changed to $newTask, model: $modelPath');
    } catch (e) {
      debugPrint('StreamingTest: Error changing task: $e');
      setState(() {
        _isModelLoading = false;
      });
    }
  }

  Widget _buildTaskSelector() {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.8),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          Text(
            'Task',
            style: TextStyle(
              color: Colors.white.withOpacity(0.7),
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),
          const SizedBox(height: 8),
          Column(
            children: YOLOTask.values.map((task) {
              final isSelected = _selectedTask == task;
              return GestureDetector(
                onTap: () => _changeTask(task),
                child: Container(
                  margin: const EdgeInsets.only(bottom: 4),
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: isSelected ? Colors.orange : Colors.transparent,
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(
                      color: isSelected ? Colors.orange : Colors.white.withOpacity(0.3),
                    ),
                  ),
                  child: Text(
                    task.name.toUpperCase(),
                    style: TextStyle(
                      color: isSelected ? Colors.white : Colors.white.withOpacity(0.8),
                      fontSize: 10,
                      fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    // Controller cleanup will be handled by YOLOView widget
    super.dispose();
  }
}