import AVFoundation
import Flutter
import UIKit

public class FLNativeView: NSObject, FlutterPlatformView, VideoCaptureDelegate {
  private let previewView: UIView
  private let videoCapture: VideoCapture
  private var busy = false
  private var currentPosition: AVCaptureDevice.Position = .back
  private weak var methodHandler: MethodCallHandler?
  private let switchCameraQueue = DispatchQueue(label: "camera.switch.queue")
  private let switchCameraSemaphore = DispatchSemaphore(value: 1)

  public init(
    frame: CGRect,
    viewIdentifier viewId: Int64,
    arguments args: Any?,
    videoCapture: VideoCapture,
    methodHandler: MethodCallHandler
  ) {
    let screenSize: CGRect = UIScreen.main.bounds
    previewView = UIView(
      frame: CGRect(x: 0, y: 0, width: screenSize.width, height: screenSize.height))

    self.videoCapture = videoCapture
    self.methodHandler = methodHandler

    super.init()

    videoCapture.nativeView = self
    videoCapture.delegate = methodHandler
    startCameraPreview(position: .back) { _ in
      // Initial camera setup complete
      print("DEBUG: Initial camera setup complete")
    }
  }

  public func view() -> UIView {
    return previewView
  }

  private func startCameraPreview(
    position: AVCaptureDevice.Position, completion: @escaping (Bool) -> Void
  ) {
    print("DEBUG: Starting camera preview with position:", position)
    videoCapture.setUp(sessionPreset: .high, position: position) { success in
      if success {
        print("DEBUG: Video capture setup completed successfully")
        if let previewLayer = self.videoCapture.previewLayer {
          DispatchQueue.main.async {
            previewLayer.frame = self.previewView.bounds
            self.previewView.layer.addSublayer(previewLayer)
            print("DEBUG: Added preview layer to view")

            self.videoCapture.start()
            print("DEBUG: Started video capture")
            self.currentPosition = position
            completion(true)
          }
        } else {
          print("DEBUG: Failed to create preview layer")
          completion(false)
        }
      } else {
        print("DEBUG: Failed to set up video capture")
        completion(false)
      }
    }
  }

  func switchCamera(completion: @escaping (Bool) -> Void) {
    print("DEBUG: switchCamera called in FLNativeView")

    switchCameraQueue.async { [weak self] in
      guard let self = self else {
        DispatchQueue.main.async { completion(false) }
        return
      }

      guard self.switchCameraSemaphore.wait(timeout: .now() + 5.0) == .success else {
        print("DEBUG: Camera switch timed out")
        DispatchQueue.main.async { completion(false) }
        return
      }

      defer { self.switchCameraSemaphore.signal() }

      if !self.busy {
        self.busy = true
        let newPosition: AVCaptureDevice.Position = self.currentPosition == .back ? .front : .back
        print("DEBUG: Switching from \(self.currentPosition) to \(newPosition)")

        DispatchQueue.main.async {
          // Stop current session
          self.videoCapture.stop()
          self.videoCapture.previewLayer?.removeFromSuperlayer()

          // Small delay to ensure cleanup
          DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            self.startCameraPreview(position: newPosition) { success in
              self.busy = false
              completion(success)
            }
          }
        }
      } else {
        print("DEBUG: Camera switch ignored - busy")
        DispatchQueue.main.async { completion(false) }
      }
    }
  }

  // MARK: - VideoCaptureDelegate
  public func videoCapture(
    _ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer
  ) {
    // Forward frames to the method handler
    methodHandler?.videoCapture(capture, didCaptureVideoFrame: sampleBuffer)
  }
}
