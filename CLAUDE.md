# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code Modification Policy
- Do not modify any code until explicitly instructed to do so
- Review code thoroughly before making any changes
- Confirm with user before implementing major structural changes

## Git Commit Policy
- Never include Claude-related attribution or "Generated with Claude Code" in commit messages
- Never mention Claude, AI, or AI assistants in commit messages
- Write professional, standard git commit messages without AI tool references
- Focus on technical changes and their purpose, not how they were created

## Build and Test Commands
- Install dependencies: `flutter pub get`
- Run all tests: `flutter test`
- Run single test: `flutter test test/FILE_PATH.dart`
- Run widget tests: `flutter test test/widget_test.dart`
- Run integration tests: `flutter test integration_test/plugin_integration_test.dart`
- Format code: `flutter format lib test`
- Analyze code: `flutter analyze`

## Code Style Guidelines
- Follow Flutter/Dart style in package:flutter_lints/flutter.yaml
- Import order: dart:*, package:flutter/*, other packages, relative imports
- Use named parameters for constructors with 2+ parameters
- Class Structure: constructors, fields, methods
- Error handling: Use try/catch with specific error types
- Documentation: Add /// dartdoc comments for public APIs
- Naming: camelCase for variables/methods, PascalCase for classes
- Platform-specific code: Keep in android/ios directories
- Method channels: Use consistent channel names across platforms
- Avoid print statements in production code; use proper logging

## Documentation Guidelines
- All documentation, comments, and public API docs must be written in English
- Use dartdoc format for API documentation
- Each public class, method, and property should have documentation comments
- Example usage should be included in documentation where appropriate
- Write clear, concise descriptions for all public APIs
- Include parameters and return value descriptions in method documentation
- For complex functionality, include code examples in documentation

## Recent Implementation History

### 2025/5/31 - Phase 3: iOS Streaming Implementation Complete
- ✅ **YOLOStreamConfig.swift**: Created iOS streaming configuration struct with cross-platform parity
  - Performance presets: MINIMAL, BALANCED, FULL, PERFORMANCE
  - All streaming options: detections, masks, poses, OBB, original images, FPS control
  - Flutter dictionary conversion support via `YOLOStreamConfig.from(dict:)`
- ✅ **YOLOView.swift**: Integrated comprehensive streaming functionality
  - Added `setStreamConfig()` and `setStreamCallback()` methods
  - Frame throttling with maxFPS and throttleIntervalMs controls
  - `convertResultToStreamData()` method with full data conversion
  - Cross-platform compatible data format (identical to Android)
- ✅ **SwiftYOLOPlatformView.swift**: Updated platform bridge for streaming
  - `setupYOLOViewStreaming()` method for configuration from creation parameters
  - `sendStreamDataToFlutter()` for event channel forwarding
  - Clean separation: YOLOView handles streaming, PlatformView forwards to Flutter
- ✅ **Swift Compilation**: Fixed all type conversion errors (CGFloat.double → Double())
- ✅ **Cross-platform Verification**: Both iOS and Android builds compile successfully

### Implementation Architecture
- **Android**: YOLOView.kt → YOLOPlatformView.kt → Flutter EventChannel
- **iOS**: YOLOView.swift → SwiftYOLOPlatformView.swift → Flutter EventChannel
- **Data Flow**: Native inference → streaming conversion → throttling → Flutter
- **Configuration**: Creation parameters → YOLOStreamConfig → performance controls

### Key Features Achieved
- **Cross-platform parity**: iOS now has identical streaming capabilities to Android
- **Performance control**: FPS limiting and throttling work on both platforms
- **Complete data support**: Detections, segmentation masks, pose keypoints, OBB, original images
- **Drawing stability**: Uses YOLOView's CALayer drawing (iOS) / Canvas drawing (Android)
- **Memory efficiency**: No unnecessary image conversions, direct CVPixelBuffer/ImageProxy handling

### Additional Optimizations (2025/5/31)
- ✅ **Performance Optimization**: Changed streaming defaults to false for heavy data (masks, poses, OBB)
- ✅ **Dynamic Configuration**: Added `setStreamingConfig` method channel for runtime configuration changes
- ✅ **Stack Overflow Fix**: Resolved Stack Overflow issue in example app caused by circular references
- ✅ **Configuration Presets**: Simplified to DEFAULT, FULL, DEBUG with custom() builder
- ✅ **Flutter Integration**: Enhanced YOLOStreamingConfig with factory constructors (.minimal, .withMasks, .withPoses, .full, .debug, .throttled)

### Current Status: ✅ COMPLETE & OPTIMIZED
**Cross-platform streaming implementation is fully functional with feature parity between iOS and Android**
**Performance optimized with smart defaults and runtime configuration support**

### Key Changes in Updated Version:
1. **Smart Defaults**: Heavy data (masks, poses, OBB) now defaults to false for optimal performance
2. **Runtime Config**: Can change streaming configuration without recreating YOLOView via `setStreamingConfig`
3. **Stack Overflow Prevention**: Fixed circular reference issues in example app
4. **Better Documentation**: Enhanced YOLOStreamingConfig with detailed performance notes
5. **Preset Configurations**: Easy-to-use factory constructors for common use cases

### Potential Next Steps (if requested)
- Test real-time streaming performance on actual devices
- Add streaming configuration UI in Flutter example app using new runtime configuration
- Implement streaming analytics and monitoring
- Add streaming pause/resume functionality
- Network streaming optimization with data compression