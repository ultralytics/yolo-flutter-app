// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Flutter
import UIKit

// Canonical Flutter 3.41 UIScene setup. Plugins are registered against the implicit engine that `FlutterSceneDelegate`
// runs (via `didInitializeImplicitFlutterEngine`), NOT against a hand-rolled engine — the previous manual engine +
// window creation in `SceneDelegate` worked under the Xcode debugger but crashed on a cold relaunch from the home
// screen. See https://flutter.dev/to/uiscene-migration.
@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
  }
}
