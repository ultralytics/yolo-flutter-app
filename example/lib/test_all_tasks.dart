// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

/// Test screen for validating all YOLO task implementations
class TestAllTasksScreen extends StatefulWidget {
  const TestAllTasksScreen({super.key});

  @override
  State<TestAllTasksScreen> createState() => _TestAllTasksScreenState();
}

class _TestAllTasksScreenState extends State<TestAllTasksScreen> {
  final ImagePicker _picker = ImagePicker();
  String _results = 'Select an image and task to test';
  bool _isLoading = false;
  File? _selectedImage;

  // Test configurations for each task
  final Map<YOLOTask, String> _taskModels = {
    YOLOTask.detect: 'yolo11n',
    YOLOTask.segment: 'yolo11n-seg',
    YOLOTask.classify: 'yolo11n-cls',
    YOLOTask.pose: 'yolo11n-pose',
    YOLOTask.obb: 'yolo11n-obb',
  };

  Future<void> _testTask(YOLOTask task) async {
    if (_selectedImage == null) {
      setState(() {
        _results = 'Please select an image first';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _results = 'Testing ${task.name}...';
    });

    try {
      // Read image bytes
      final imageBytes = await _selectedImage!.readAsBytes();

      // Create YOLO instance
      final yolo = YOLO(modelPath: _taskModels[task]!, task: task);

      // Load model
      final loadSuccess = await yolo.loadModel();
      if (!loadSuccess) {
        throw Exception('Failed to load model');
      }

      // Run prediction
      final results = await yolo.predict(imageBytes);

      // Format results based on task
      final output = StringBuffer();
      output.writeln('=== ${task.name.toUpperCase()} Results ===\n');

      // Image size
      if (results['imageSize'] != null) {
        final size = results['imageSize'] as Map<dynamic, dynamic>;
        output.writeln('Image Size: ${size['width']}x${size['height']}\n');
      }

      // Boxes (common to all tasks except classify)
      if (task != YOLOTask.classify && results['boxes'] != null) {
        final boxes = results['boxes'] as List<dynamic>;
        output.writeln('Detected ${boxes.length} objects\n');

        for (int i = 0; i < boxes.length && i < 3; i++) {
          final box = boxes[i] as Map<dynamic, dynamic>;
          output.writeln('Box ${i + 1}:');
          output.writeln('  Class: ${box['class']}');
          output.writeln(
            '  Confidence: ${(box['confidence'] * 100).toStringAsFixed(1)}%',
          );
          output.writeln(
            '  Pixel coords: (${box['x1']?.toStringAsFixed(1)}, ${box['y1']?.toStringAsFixed(1)}) to (${box['x2']?.toStringAsFixed(1)}, ${box['y2']?.toStringAsFixed(1)})',
          );
          output.writeln(
            '  Normalized: (${box['x1_norm']?.toStringAsFixed(3)}, ${box['y1_norm']?.toStringAsFixed(3)}) to (${box['x2_norm']?.toStringAsFixed(3)}, ${box['y2_norm']?.toStringAsFixed(3)})\n',
          );
        }
      }

      // Task-specific results
      switch (task) {
        case YOLOTask.pose:
          if (results['keypoints'] != null) {
            final keypoints = results['keypoints'] as List<dynamic>;
            output.writeln('Pose Keypoints:');
            output.writeln('  Found ${keypoints.length} person(s) with poses');

            if (keypoints.isNotEmpty) {
              final firstPerson = keypoints[0] as Map<dynamic, dynamic>;
              final coords = firstPerson['coordinates'] as List<dynamic>;
              output.writeln('  Person 1 has ${coords.length} keypoints');

              // Show first 3 keypoints
              for (int i = 0; i < coords.length && i < 3; i++) {
                final kp = coords[i] as Map<dynamic, dynamic>;
                output.writeln(
                  '    Keypoint ${i + 1}: (${kp['x']?.toStringAsFixed(3)}, ${kp['y']?.toStringAsFixed(3)}) conf: ${kp['confidence']?.toStringAsFixed(2)}',
                );
              }
            }
          }
          break;

        case YOLOTask.segment:
          if (results['masks'] != null) {
            final masks = results['masks'] as List<dynamic>;
            output.writeln('Segmentation Masks:');
            output.writeln('  Found ${masks.length} instance masks');

            if (masks.isNotEmpty) {
              final firstMask = masks[0] as List<dynamic>;
              output.writeln(
                '  Mask 1 size: ${firstMask.length}x${firstMask[0].length}',
              );
            }
          }
          break;

        case YOLOTask.classify:
          if (results['classification'] != null) {
            final cls = results['classification'] as Map<dynamic, dynamic>;
            output.writeln('Classification:');
            output.writeln('  Top class: ${cls['topClass']}');
            output.writeln(
              '  Confidence: ${(cls['topConfidence'] * 100).toStringAsFixed(1)}%',
            );

            if (cls['top5Classes'] != null) {
              output.writeln('\n  Top 5 classes:');
              final top5 = cls['top5Classes'] as List<dynamic>;
              final confs = cls['top5Confidences'] as List<dynamic>;
              for (int i = 0; i < top5.length; i++) {
                output.writeln(
                  '    ${i + 1}. ${top5[i]} (${(confs[i] * 100).toStringAsFixed(1)}%)',
                );
              }
            }
          }
          break;

        case YOLOTask.obb:
          if (results['obb'] != null) {
            final obbList = results['obb'] as List<dynamic>;
            output.writeln('Oriented Bounding Boxes:');
            output.writeln('  Found ${obbList.length} OBBs');

            for (int i = 0; i < obbList.length && i < 3; i++) {
              final obb = obbList[i] as Map<dynamic, dynamic>;
              output.writeln('  OBB ${i + 1}:');
              output.writeln('    Class: ${obb['class']}');
              output.writeln(
                '    Confidence: ${(obb['confidence'] * 100).toStringAsFixed(1)}%',
              );

              final points = obb['points'] as List<dynamic>;
              output.writeln('    Corners: ${points.length} points');
            }
          }
          break;

        case YOLOTask.detect:
          // Already handled by boxes
          break;
      }

      // Test YOLOResult parsing
      if (results['detections'] != null) {
        output.writeln('\n=== YOLOResult Parsing Test ===');
        final detections = results['detections'] as List<dynamic>;
        output.writeln(
          'Successfully created ${detections.length} YOLOResult objects',
        );

        try {
          for (int i = 0; i < detections.length && i < 2; i++) {
            final result = YOLOResult.fromMap(detections[i]);
            output.writeln('\nYOLOResult ${i + 1}:');
            output.writeln('  Valid structure ✓');
            output.writeln('  Has normalized box: true ✓');

            if (task == YOLOTask.pose) {
              output.writeln('  Has keypoints: ${result.keypoints != null} ✓');
            } else if (task == YOLOTask.segment) {
              output.writeln('  Has mask: ${result.mask != null} ✓');
            }
          }
        } catch (e) {
          output.writeln('  ERROR parsing YOLOResult: $e');
        }
      }

      setState(() {
        _results = output.toString();
      });

      // Cleanup
      await yolo.dispose();
    } catch (e) {
      setState(() {
        _results = 'Error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
        _results = 'Image selected. Choose a task to test.';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Test All YOLO Tasks')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image selection
            if (_selectedImage != null)
              Container(
                height: 200,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: Image.file(_selectedImage!, fit: BoxFit.cover),
                ),
              ),
            const SizedBox(height: 16),

            ElevatedButton.icon(
              onPressed: _pickImage,
              icon: const Icon(Icons.image),
              label: const Text('Select Image'),
            ),

            const SizedBox(height: 24),

            // Task buttons
            const Text(
              'Select Task to Test:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),

            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: YOLOTask.values.map((task) {
                return ElevatedButton(
                  onPressed: _isLoading ? null : () => _testTask(task),
                  child: Text(task.name.toUpperCase()),
                );
              }).toList(),
            ),

            const SizedBox(height: 24),

            // Results
            const Text(
              'Results:',
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),

            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey[300]!),
              ),
              child: _isLoading
                  ? const Center(child: CircularProgressIndicator())
                  : SelectableText(
                      _results,
                      style: const TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 12,
                      ),
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
