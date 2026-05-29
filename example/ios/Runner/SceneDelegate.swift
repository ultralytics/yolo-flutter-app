// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Flutter
import UIKit

/// Canonical Flutter 3.41 scene delegate. `FlutterSceneDelegate` loads the `Main` storyboard (a `FlutterViewController`)
/// against the implicit engine, whose plugins are registered in `AppDelegate.didInitializeImplicitFlutterEngine`. Do
/// NOT hand-roll a `FlutterEngine`/`UIWindow` here or call `super.scene(willConnectTo:)` on top of a manual setup —
/// that double-configured the scene and crashed on a cold relaunch from the home screen.
/// See https://flutter.dev/to/uiscene-migration.
@objc class SceneDelegate: FlutterSceneDelegate {
}
