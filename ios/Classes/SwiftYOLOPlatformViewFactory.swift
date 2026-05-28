// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

// See YOLOPlugin.swift — `@preconcurrency` keeps Flutter's pre-concurrency types Sendable-clean under Swift 6
// strict concurrency.
@preconcurrency import Flutter
import UIKit

// Thread-safe view registry
private class YOLOViewRegistry {
  private var _views: [Int: YOLOView] = [:]
  private let lock = NSLock()

  func get(for viewId: Int) -> YOLOView? {
    lock.lock()
    defer { lock.unlock() }
    return _views[viewId]
  }

  func set(_ view: YOLOView?, for viewId: Int) {
    lock.lock()
    defer { lock.unlock() }
    _views[viewId] = view
  }

  func remove(for viewId: Int) {
    lock.lock()
    defer { lock.unlock() }
    _views.removeValue(forKey: viewId)
  }
}

// See YOLOPlugin.swift — `@preconcurrency` on the conformance keeps the FlutterPlatformViewFactory (non-isolated)
// protocol from tripping Swift 6 strict-isolation warnings on this `@MainActor` class.
@MainActor
public final class SwiftYOLOPlatformViewFactory: NSObject, @preconcurrency FlutterPlatformViewFactory {
  private var messenger: FlutterBinaryMessenger
  // YOLOViewRegistry is internally thread-safe (NSLock around its dictionary), so `nonisolated(unsafe)` lets the
  // factory's `nonisolated` accessors (e.g. unregisterSync) reach it from background contexts during shutdown
  // without tripping Swift-6 "main actor-isolated property" errors.
  nonisolated(unsafe) private static let viewRegistry = YOLOViewRegistry()

  init(messenger: FlutterBinaryMessenger) {
    self.messenger = messenger
    super.init()
  }

  static func getYOLOView(for viewId: Int) -> YOLOView? {
    return viewRegistry.get(for: viewId)
  }

  static func register(_ yoloView: YOLOView, for viewId: Int) {
    viewRegistry.set(yoloView, for: viewId)
  }

  static func unregister(for viewId: Int) {
    viewRegistry.remove(for: viewId)
  }

  nonisolated static func unregisterSync(for viewId: Int) {
    viewRegistry.remove(for: viewId)
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
