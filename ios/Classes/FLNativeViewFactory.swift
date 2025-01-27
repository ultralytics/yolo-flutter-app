//
//  FLNativeViewFactory.swift
//  ultralytics_yolo
//
//  Created by Sergio Sánchez on 9/11/23.
//

import Flutter

public class FLNativeViewFactory: NSObject, FlutterPlatformViewFactory {
  private let videoCapture: VideoCapture
  private let methodHandler: MethodCallHandler

  public init(videoCapture: VideoCapture, methodHandler: MethodCallHandler) {
    self.videoCapture = videoCapture
    self.methodHandler = methodHandler
    super.init()
  }

  public func create(
    withFrame frame: CGRect,
    viewIdentifier viewId: Int64,
    arguments args: Any?
  ) -> FlutterPlatformView {
    return FLNativeView(
      frame: frame,
      viewIdentifier: viewId,
      arguments: args,
      videoCapture: videoCapture,
      methodHandler: methodHandler
    )
  }
}
