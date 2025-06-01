# Examples & Use Cases

This guide showcases common patterns and real-world applications using YOLO Flutter.

## 🎯 Object Detection Examples

### Basic Object Detection

```dart
YOLOView(
  modelPath: 'assets/yolo11n.tflite',
  task: YOLOTask.detect,
  onResult: (results) {
    for (final result in results) {
      print('${result.className}: ${result.confidence}');
    }
  },
)
```

### Security Camera App

```dart
class SecurityCamera extends StatefulWidget {
  @override
  _SecurityCameraState createState() => _SecurityCameraState();
}

class _SecurityCameraState extends State<SecurityCamera> {
  List<String> _alerts = [];
  final Set<String> _securityTargets = {'person', 'car', 'truck'};

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Security Monitor')),
      body: Column(
        children: [
          // Alert panel
          Container(
            height: 100,
            color: _alerts.isNotEmpty ? Colors.red[100] : Colors.green[100],
            child: ListView.builder(
              itemCount: _alerts.length,
              itemBuilder: (context, index) => ListTile(
                leading: Icon(Icons.warning, color: Colors.red),
                title: Text(_alerts[index]),
              ),
            ),
          ),
          // Camera view
          Expanded(
            child: YOLOView(
              modelPath: 'assets/yolo11n.tflite',
              task: YOLOTask.detect,
              confidenceThreshold: 0.7, // High confidence for security
              onResult: (results) {
                final detectedTargets = results
                    .where((r) => _securityTargets.contains(r.className))
                    .map((r) => r.className)
                    .toSet();
                
                if (detectedTargets.isNotEmpty) {
                  setState(() {
                    _alerts.add(
                      '${DateTime.now().toString().substring(11, 19)}: '
                      'Detected ${detectedTargets.join(", ")}'
                    );
                    if (_alerts.length > 10) _alerts.removeAt(0);
                  });
                }
              },
            ),
          ),
        ],
      ),
    );
  }
}
```

### Inventory Management

```dart
class InventoryScanner extends StatefulWidget {
  @override
  _InventoryScannerState createState() => _InventoryScannerState();
}

class _InventoryScannerState extends State<InventoryScanner> {
  Map<String, int> _inventory = {};

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Inventory Scanner')),
      body: Row(
        children: [
          // Camera view
          Expanded(
            flex: 2,
            child: YOLOView(
              modelPath: 'assets/yolo11n.tflite',
              task: YOLOTask.detect,
              onResult: (results) {
                setState(() {
                  _inventory.clear();
                  for (final result in results) {
                    _inventory[result.className] = 
                        (_inventory[result.className] ?? 0) + 1;
                  }
                });
              },
            ),
          ),
          // Inventory list
          Expanded(
            flex: 1,
            child: Container(
              color: Colors.grey[100],
              child: Column(
                children: [
                  Padding(
                    padding: EdgeInsets.all(8),
                    child: Text('Current Inventory', 
                        style: Theme.of(context).textTheme.headlineSmall),
                  ),
                  Expanded(
                    child: ListView.builder(
                      itemCount: _inventory.length,
                      itemBuilder: (context, index) {
                        final item = _inventory.keys.elementAt(index);
                        final count = _inventory[item]!;
                        return ListTile(
                          title: Text(item),
                          trailing: CircleAvatar(
                            child: Text('$count'),
                            backgroundColor: Colors.blue,
                          ),
                        );
                      },
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
```

## 🎭 Segmentation Examples

### Photo Editor with Object Isolation

```dart
class PhotoEditor extends StatefulWidget {
  @override
  _PhotoEditorState createState() => _PhotoEditorState();
}

class _PhotoEditorState extends State<PhotoEditor> {
  List<YOLOResult> _currentResults = [];
  String? _selectedObject;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Photo Editor')),
      body: Column(
        children: [
          // Object selector
          Container(
            height: 60,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemCount: _currentResults.length,
              itemBuilder: (context, index) {
                final result = _currentResults[index];
                final isSelected = _selectedObject == result.className;
                
                return Padding(
                  padding: EdgeInsets.all(4),
                  child: FilterChip(
                    label: Text(result.className),
                    selected: isSelected,
                    onSelected: (selected) {
                      setState(() {
                        _selectedObject = selected ? result.className : null;
                      });
                    },
                  ),
                );
              },
            ),
          ),
          // Camera with segmentation
          Expanded(
            child: YOLOView(
              modelPath: 'assets/yolo11n-seg.tflite',
              task: YOLOTask.segment,
              streamingConfig: YOLOStreamingConfig.withMasks(),
              onResult: (results) {
                setState(() {
                  _currentResults = results;
                });
              },
            ),
          ),
          // Edit controls
          Container(
            padding: EdgeInsets.all(16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  icon: Icon(Icons.blur_on),
                  label: Text('Blur Background'),
                  onPressed: _selectedObject != null ? () {
                    // Implement background blur using mask data
                  } : null,
                ),
                ElevatedButton.icon(
                  icon: Icon(Icons.color_lens),
                  label: Text('Change Colors'),
                  onPressed: _selectedObject != null ? () {
                    // Implement color change using mask data
                  } : null,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
```

## 🤸 Pose Estimation Examples

### Fitness App with Pose Tracking

```dart
class FitnessTracker extends StatefulWidget {
  @override
  _FitnessTrackerState createState() => _FitnessTrackerState();
}

class _FitnessTrackerState extends State<FitnessTracker> {
  int _squatCount = 0;
  bool _isSquatPosition = false;
  List<Point> _lastKeypoints = [];

  bool _isInSquatPosition(List<Point> keypoints) {
    // Simplified squat detection logic
    // In real app, you'd use proper pose analysis
    if (keypoints.length < 17) return false;
    
    final leftKnee = keypoints[13];  // Left knee
    final rightKnee = keypoints[14]; // Right knee
    final leftHip = keypoints[11];   // Left hip
    final rightHip = keypoints[12];  // Right hip
    
    // Check if knees are significantly below hips
    final avgKneeY = (leftKnee.y + rightKnee.y) / 2;
    final avgHipY = (leftHip.y + rightHip.y) / 2;
    
    return avgKneeY > avgHipY + 50; // Threshold for squat position
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Squat Counter')),
      body: Column(
        children: [
          // Stats panel
          Container(
            padding: EdgeInsets.all(20),
            color: Colors.blue[50],
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                Column(
                  children: [
                    Text('$_squatCount', 
                        style: TextStyle(fontSize: 32, fontWeight: FontWeight.bold)),
                    Text('Squats'),
                  ],
                ),
                Column(
                  children: [
                    Icon(
                      _isSquatPosition ? Icons.fitness_center : Icons.accessibility,
                      size: 32,
                      color: _isSquatPosition ? Colors.green : Colors.grey,
                    ),
                    Text(_isSquatPosition ? 'Squat Position' : 'Standing'),
                  ],
                ),
              ],
            ),
          ),
          // Pose detection view
          Expanded(
            child: YOLOView(
              modelPath: 'assets/yolo11n-pose.tflite',
              task: YOLOTask.pose,
              streamingConfig: YOLOStreamingConfig.withPoses(),
              onResult: (results) {
                if (results.isNotEmpty && results.first.keypoints != null) {
                  final keypoints = results.first.keypoints!;
                  final currentlyInSquat = _isInSquatPosition(keypoints);
                  
                  // Count squat completion (transition from squat to standing)
                  if (_isSquatPosition && !currentlyInSquat) {
                    setState(() {
                      _squatCount++;
                    });
                  }
                  
                  setState(() {
                    _isSquatPosition = currentlyInSquat;
                    _lastKeypoints = keypoints;
                  });
                }
              },
            ),
          ),
          // Reset button
          Padding(
            padding: EdgeInsets.all(16),
            child: ElevatedButton(
              onPressed: () {
                setState(() {
                  _squatCount = 0;
                });
              },
              child: Text('Reset Counter'),
            ),
          ),
        ],
      ),
    );
  }
}
```

## 🏷️ Classification Examples

### Content Moderation

```dart
class ContentModerator extends StatefulWidget {
  @override
  _ContentModeratorState createState() => _ContentModeratorState();
}

class _ContentModeratorState extends State<ContentModerator> {
  String _contentStatus = 'Scanning...';
  Color _statusColor = Colors.grey;
  final Set<String> _inappropriateContent = {
    'weapon', 'violence', 'adult_content'
  };

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Content Moderator')),
      body: Column(
        children: [
          // Status indicator
          Container(
            width: double.infinity,
            padding: EdgeInsets.all(20),
            color: _statusColor.withOpacity(0.1),
            child: Column(
              children: [
                Icon(
                  _statusColor == Colors.green ? Icons.check_circle : 
                  _statusColor == Colors.red ? Icons.error : Icons.hourglass_empty,
                  size: 48,
                  color: _statusColor,
                ),
                SizedBox(height: 8),
                Text(
                  _contentStatus,
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: _statusColor,
                  ),
                ),
              ],
            ),
          ),
          // Camera view
          Expanded(
            child: YOLOView(
              modelPath: 'assets/content_classifier.tflite', // Custom model
              task: YOLOTask.classify,
              confidenceThreshold: 0.8,
              onResult: (results) {
                if (results.isNotEmpty) {
                  final topResult = results.first;
                  final isInappropriate = _inappropriateContent
                      .contains(topResult.className.toLowerCase());
                  
                  setState(() {
                    if (isInappropriate) {
                      _contentStatus = 'Content Blocked';
                      _statusColor = Colors.red;
                    } else {
                      _contentStatus = 'Content Approved';
                      _statusColor = Colors.green;
                    }
                  });
                }
              },
            ),
          ),
        ],
      ),
    );
  }
}
```

## 📦 OBB Detection Examples

### Document Scanner

```dart
class DocumentScanner extends StatefulWidget {
  @override
  _DocumentScannerState createState() => _DocumentScannerState();
}

class _DocumentScannerState extends State<DocumentScanner> {
  List<Map<String, dynamic>> _documents = [];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Document Scanner')),
      body: Column(
        children: [
          // Document list
          Container(
            height: 120,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemCount: _documents.length,
              itemBuilder: (context, index) {
                final doc = _documents[index];
                return Card(
                  margin: EdgeInsets.all(8),
                  child: Container(
                    width: 100,
                    padding: EdgeInsets.all(8),
                    child: Column(
                      children: [
                        Icon(Icons.description, size: 32),
                        Text(doc['type']),
                        Text('${doc['angle']}°', 
                            style: TextStyle(fontSize: 12)),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
          // Camera view
          Expanded(
            child: YOLOView(
              modelPath: 'assets/document_obb.tflite', // Custom OBB model
              task: YOLOTask.obb,
              streamingConfig: YOLOStreamingConfig.custom(includeOBB: true),
              onResult: (results) {
                setState(() {
                  _documents.clear();
                  for (final result in results) {
                    // OBB data would be available in streaming mode
                    _documents.add({
                      'type': result.className,
                      'confidence': result.confidence,
                      'angle': 0, // Would come from OBB data
                    });
                  }
                });
              },
            ),
          ),
        ],
      ),
    );
  }
}
```

## 🔧 Performance Examples

### Adaptive Quality Based on Device

```dart
class AdaptiveYOLOView extends StatefulWidget {
  @override
  _AdaptiveYOLOViewState createState() => _AdaptiveYOLOViewState();
}

class _AdaptiveYOLOViewState extends State<AdaptiveYOLOView> {
  late YOLOStreamingConfig _config;
  double _currentFps = 0;

  @override
  void initState() {
    super.initState();
    _config = _getOptimalConfig();
  }

  YOLOStreamingConfig _getOptimalConfig() {
    // Detect device performance and adapt
    // This is a simplified example
    final isHighEnd = _isHighEndDevice();
    
    if (isHighEnd) {
      return YOLOStreamingConfig.custom(
        maxFPS: 30,
        inferenceFrequency: 25,
        includeMasks: true,
        includePoses: true,
      );
    } else {
      return YOLOStreamingConfig.custom(
        maxFPS: 15,
        inferenceFrequency: 10,
        includeMasks: false,
        includePoses: false,
      );
    }
  }

  bool _isHighEndDevice() {
    // Implement device detection logic
    // Could check available RAM, CPU cores, etc.
    return true; // Placeholder
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Adaptive YOLO'),
        actions: [
          Center(
            child: Padding(
              padding: EdgeInsets.all(8),
              child: Text('${_currentFps.toStringAsFixed(1)} FPS'),
            ),
          ),
        ],
      ),
      body: YOLOView(
        modelPath: 'assets/yolo11n.tflite',
        task: YOLOTask.detect,
        streamingConfig: _config,
        onPerformanceMetrics: (metrics) {
          setState(() {
            _currentFps = metrics.fps;
          });
          
          // Auto-adjust if performance is poor
          if (metrics.fps < 10 && _config.maxFPS! > 10) {
            setState(() {
              _config = YOLOStreamingConfig.custom(
                maxFPS: 10,
                inferenceFrequency: 5,
              );
            });
          }
        },
        onResult: (results) {
          // Handle results
        },
      ),
    );
  }
}
```

## 🚀 Next Steps

- **[Performance Optimization](./performance.md)** - Advanced performance tuning
- **[Streaming Guide](./streaming.md)** - Real-time processing details  
- **[API Reference](./api-reference.md)** - Complete technical documentation
- **[Troubleshooting](./troubleshooting.md)** - Common issues and solutions

## Example Apps

Check out complete example applications:
- **[Basic Example](../example/)** - Simple detection app
- **[Streaming Demo](../streaming_test_example/)** - Advanced real-time features
- **[Simple Example](../simple_example/)** - Minimal implementation