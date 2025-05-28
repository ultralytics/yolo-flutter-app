// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Flutter
import UIKit

@MainActor
public class SwiftYoloPlatformViewFactory: NSObject, FlutterPlatformViewFactory {
  private var messenger: FlutterBinaryMessenger
  static var yoloViews: [Int: YOLOView] = [:]

  init(messenger: FlutterBinaryMessenger) {
    self.messenger = messenger
    super.init()
  }

  static func getYoloView(for viewId: Int) -> YOLOView? {
    return yoloViews[viewId]
  }

  static func register(_ yoloView: YOLOView, for viewId: Int) {
    yoloViews[viewId] = yoloView
  }

  static func unregister(for viewId: Int) {
    yoloViews.removeValue(forKey: viewId)
  }

  public func createArgsCodec() -> FlutterMessageCodec & NSObjectProtocol {
    // Dart 側で `creationParamsCodec: const StandardMessageCodec()` を指定しているので
    // こちらでも FlutterStandardMessageCodec.sharedInstance() を使う
    return FlutterStandardMessageCodec.sharedInstance()
  }

  public func create(
    withFrame frame: CGRect,
    viewIdentifier viewId: Int64,
    arguments args: Any?
  ) -> FlutterPlatformView {
    return SwiftYoloPlatformView(
      frame: frame,
      viewId: viewId,
      args: args,
      messenger: messenger
    )
  }
}
