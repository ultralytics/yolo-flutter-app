import Flutter
import UIKit

public class SwiftUltralyticsYoloPlugin: NSObject, FlutterPlugin {
  // Keep strong references to prevent deallocation
  private static var methodHandler: MethodCallHandler?
  private static var videoCapture: VideoCapture?

  public static func register(with registrar: FlutterPluginRegistrar) {
    // Create VideoCapture instance
    videoCapture = VideoCapture()

    // Create MethodCallHandler instance
    methodHandler = MethodCallHandler(
      binaryMessenger: registrar.messenger(),
      videoCapture: videoCapture!
    )

    // Register method channel
    let channel = FlutterMethodChannel(
      name: "ultralytics_yolo",
      binaryMessenger: registrar.messenger()
    )
    channel.setMethodCallHandler(methodHandler!.handle)

    // Register native view factory with both dependencies
    registrar.register(
      FLNativeViewFactory(
        videoCapture: videoCapture!,
        methodHandler: methodHandler!
      ),
      withId: "ultralytics_yolo_camera_preview"
    )
  }
}
