# CHANGELOG Draft - Multi-Instance Support Feature

This file contains changelog entries for the multi-instance support feature that will be added to CHANGELOG.md when the feature is officially released.

## 0.1.17

- Add comprehensive multi-instance usage documentation with practical examples
- Fix pub.dev example visibility by removing `publish_to: "none"` from example/pubspec.yaml
- Improve package compatibility with relaxed SDK constraints (>=3.0.0 <4.0.0)
- Include dual model comparison and performance monitoring examples
- Fix iOS Swift compilation error by removing unsupported @preconcurrency attribute for Xcode 15.x compatibility

## 0.1.16

- Add multi-instance YOLO support for running multiple models simultaneously
- Implement backward compatibility with `useMultiInstance` parameter (defaults to false)
- Add unique instance ID generation and management for iOS and Android
- Fix iOS Swift compilation errors and Android initialization order issues
- Create YOLOInstanceManager for platform-specific multi-instance handling
- Resolve MissingPluginException through proper backward compatibility
- Add comprehensive error handling and proper disposal patterns

---

**Note:** These entries should be moved to the main CHANGELOG.md when the multi-instance feature is officially released.
