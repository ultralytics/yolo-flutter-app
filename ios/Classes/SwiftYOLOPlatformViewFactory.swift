// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Flutter
import UIKit

@MainActor
public class SwiftYOLOPlatformViewFactory: NSObject, FlutterPlatformViewFactory {
  private var messenger: FlutterBinaryMessenger
  static var yoloViews: [Int: YOLOView] = [:]

  init(messenger: FlutterBinaryMessenger) {
    self.messenger = messenger
    super.init()
  }

  static func getYOLOView(for viewId: Int) -> YOLOView? {
    return yoloViews[viewId]
  }

  static func register(_ yoloView: YOLOView, for viewId: Int) {
    yoloViews[viewId] = yoloView
  }

  static func unregister(for viewId: Int) {
    yoloViews.removeValue(forKey: viewId)
  }

  public func createArgsCodec() -> FlutterMessageCodec & NSObjectProtocol {
    return FlutterStandardMessageCodec.sharedInstance()
  }

  public func create(
    withFrame frame: CGRect,
    viewIdentifier viewId: Int64,
    arguments args: Any?
  ) -> FlutterPlatformView {
    return SwiftYOLOPlatformView(
      frame: frame,
      viewId: viewId,
      args: args,
      messenger: messenger
    )
  }
}
