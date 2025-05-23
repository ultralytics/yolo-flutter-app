# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Modification Policy

- Do not modify any code until explicitly instructed to do so
- Review code thoroughly before making any changes
- Confirm with user before implementing major structural changes

## Git Commit Messages

- Do not include any Claude-related attribution or references in commit messages
- Keep commit messages professional and focused on the changes made
- Do not add "Generated with Claude Code" or "Co-Authored-By: Claude" signatures

## Build and Test Commands

- Install dependencies: `flutter pub get`
- Run all tests: `flutter test`
- Run single test: `flutter test test/FILE_PATH.dart`
- Format code: `flutter format lib test`
- Analyze code: `dart analyze --fatal-warnings`
- Code coverage: `flutter test --coverage`

## Architecture Overview

This repository contains a Flutter plugin for integrating Ultralytics YOLO computer vision models into mobile applications. The plugin follows a platform channel architecture:

1. **Core Plugin Structure**:

   - `lib/yolo.dart`: Main plugin class
   - `lib/yolo_view.dart`: Camera preview widget with YOLO detection
   - `lib/yolo_task.dart`: Enum defining supported YOLO tasks
   - `lib/yolo_result.dart`: Detection result classes

2. **Platform Channel Communication**:

   - Method channels for one-time operations (loadModel, detect, classify)
   - Event channels for streaming data (inferenceTime, fpsRate, results)

3. **Native Implementation**:
   - Android: Uses TensorFlow Lite for model inference
   - iOS: Uses Core ML for model inference

## Code Style Guidelines

- Follow Flutter/Dart style in package:flutter_lints/flutter.yaml
- Import order: dart:_, package:flutter/_, other packages, relative imports
- Use named parameters for constructors with 2+ parameters
- Class Structure: constructors, fields, methods
- Error handling: Use try/catch with specific error types
- Documentation: Add /// dartdoc comments for public APIs
- Naming: camelCase for variables/methods, PascalCase for classes
- Platform-specific code: Keep in android/ios directories
- Method channels: Use consistent channel names across platforms
- Avoid print statements in production code; use proper logging
- Use English only for all code comments and documentation (no Japanese or other languages)

## Important Implementation Details

1. **Model Format Support**:

   - Android: TensorFlow Lite (.tflite) models
   - iOS: Core ML (.mlpackage, .mlmodel) models

2. **Supported YOLO Tasks**:

   - Object Detection
   - Image Segmentation
   - Image Classification
   - Pose Estimation
   - Oriented Bounding Boxes (OBB)

3. **Permission Handling**:

   - Camera permissions required for live detection
   - Storage permissions required for image-based detection

4. **Performance Considerations**:
   - Native code handles all intensive processing
   - Threshold parameters can be adjusted for performance optimization
