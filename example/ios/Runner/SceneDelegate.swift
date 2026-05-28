// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Flutter
import UIKit

/// iOS 26+ requires a UISceneDelegate. We create the FlutterEngine here (instead of letting the storyboard implicitly
/// spin up a fresh one) and register the GeneratedPluginRegistrant against it — without this every plugin call lands
/// on an engine with no plugins and Dart sees `MissingPluginException` / `channel-error` for everything (wakelock_plus,
/// shared_preferences, ultralytics_yolo, ...). Subclassing Flutter's own delegate keeps its scene-lifecycle plumbing.
/// See https://flutter.dev/to/uiscene-migration.
@objc class SceneDelegate: FlutterSceneDelegate {
  var flutterEngine: FlutterEngine?

  override func scene(
    _ scene: UIScene,
    willConnectTo session: UISceneSession,
    options connectionOptions: UIScene.ConnectionOptions
  ) {
    guard let windowScene = scene as? UIWindowScene else {
      super.scene(scene, willConnectTo: session, options: connectionOptions)
      return
    }

    let engine = FlutterEngine(name: "io.ultralytics.yoloExample")
    engine.run()
    GeneratedPluginRegistrant.register(with: engine)
    self.flutterEngine = engine

    let viewController = FlutterViewController(engine: engine, nibName: nil, bundle: nil)
    let window = UIWindow(windowScene: windowScene)
    window.rootViewController = viewController
    self.window = window
    window.makeKeyAndVisible()

    super.scene(scene, willConnectTo: session, options: connectionOptions)
  }
}
