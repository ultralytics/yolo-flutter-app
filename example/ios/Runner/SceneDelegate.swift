// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Flutter
import UIKit

/// iOS 26+ requires a UISceneDelegate. Subclasses Flutter's own delegate (so its plugin lifecycle is preserved), then
/// instantiates the storyboard's initial FlutterViewController and attaches it to the windowScene — Flutter's stock
/// `FlutterSceneDelegate` doesn't load the storyboard root on its own, which leaves the app on a white screen.
/// Mirrors the migration pattern from https://flutter.dev/to/uiscene-migration.
@objc class SceneDelegate: FlutterSceneDelegate {
  override func scene(
    _ scene: UIScene,
    willConnectTo session: UISceneSession,
    options connectionOptions: UIScene.ConnectionOptions
  ) {
    guard let windowScene = scene as? UIWindowScene else {
      super.scene(scene, willConnectTo: session, options: connectionOptions)
      return
    }
    let window = UIWindow(windowScene: windowScene)
    let storyboard = UIStoryboard(name: "Main", bundle: nil)
    window.rootViewController = storyboard.instantiateInitialViewController()
    self.window = window
    window.makeKeyAndVisible()
    super.scene(scene, willConnectTo: session, options: connectionOptions)
  }
}
