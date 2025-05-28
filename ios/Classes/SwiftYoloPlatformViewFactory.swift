// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import Flutter
import UIKit

@MainActor
public class SwiftYOLOPlatformViewFactory: NSObject, FlutterPlatformViewFactory {
  private var messenger: FlutterBinaryMessenger

  init(messenger: FlutterBinaryMessenger) {
    self.messenger = messenger
    super.init()
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
    return SwiftYOLOPlatformView(
      frame: frame,
      viewId: viewId,
      args: args,
      messenger: messenger
    )
  }
}
