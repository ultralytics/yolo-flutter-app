// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Flutter
import UIKit

@MainActor
public class SwiftYOLOPlatformViewFactory: NSObject, FlutterPlatformViewFactory {
  private var messenger: FlutterBinaryMessenger
  nonisolated static var yoloViews: [Int: YOLOView] = [:]
  nonisolated static let yoloViewsLock = NSLock()

  init(messenger: FlutterBinaryMessenger) {
    self.messenger = messenger
    super.init()
  }

  nonisolated static func getYOLOView(for viewId: Int) -> YOLOView? {
    yoloViewsLock.lock()
    defer { yoloViewsLock.unlock() }
    return yoloViews[viewId]
  }

  nonisolated static func register(_ yoloView: YOLOView, for viewId: Int) {
    yoloViewsLock.lock()
    defer { yoloViewsLock.unlock() }
    yoloViews[viewId] = yoloView
  }

  nonisolated static func unregister(for viewId: Int) {
    yoloViewsLock.lock()
    defer { yoloViewsLock.unlock() }
    yoloViews.removeValue(forKey: viewId)
  }

  nonisolated static func unregisterSync(for viewId: Int) {
    yoloViewsLock.lock()
    defer { yoloViewsLock.unlock() }
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
