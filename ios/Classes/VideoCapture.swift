// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Ultralytics YOLO Package, managing camera capture for real-time inference.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The VideoCapture component manages the camera and video processing pipeline for real-time
//  object detection. It handles setting up the AVCaptureSession, managing camera devices,
//  configuring camera properties like focus and exposure, and processing video frames for
//  model inference. The class delivers capture frames to the predictor component for real-time
//  analysis and returns results through delegate callbacks. It also supports camera controls
//  such as switching between front and back cameras, zooming, and capturing still photos.

import AVFoundation
import CoreVideo
import UIKit

/// Protocol for receiving video capture frame processing results.
@MainActor
protocol VideoCaptureDelegate: AnyObject {
  func onPredict(result: YOLOResult)
  func onInferenceTime(speed: Double, fps: Double)
}

func bestCaptureDevice(position: AVCaptureDevice.Position) -> AVCaptureDevice? {
  // Prefer the virtual multi-camera so the ultra-wide (0.5x) is a constituent of the active device and reachable via
  // videoZoomFactor — a plain `.builtInWideAngleCamera`/`.builtInDualCamera` has no ultra-wide, so 0.5x is impossible
  // and only zoom-in (2x/4x) works. Mirrors `yolo-ios-app/Sources/YOLO/VideoCapture.swift#bestCaptureDevice`.
  let preferredTypes: [AVCaptureDevice.DeviceType] =
    position == .back
    ? [.builtInTripleCamera, .builtInDualWideCamera, .builtInDualCamera, .builtInWideAngleCamera]
    : [.builtInTrueDepthCamera, .builtInWideAngleCamera]

  for deviceType in preferredTypes {
    if let device = AVCaptureDevice.default(deviceType, for: .video, position: position) {
      return device
    }
  }

  return nil
}

/// Converts a raw `videoZoomFactor` into the user-facing factor shown in the UI (e.g. raw 1.0 on a device whose widest
/// lens is the ultra-wide reads as 0.5x). iOS 18+ exposes the per-device multiplier; earlier OSes have no sub-1x
/// display concept so the raw value is shown as-is. Mirrors `yolo-ios-app/Sources/YOLO/VideoCapture.swift`.
func displayZoomFactor(_ zoomFactor: CGFloat, for device: AVCaptureDevice) -> CGFloat {
  if #available(iOS 18.0, *) {
    return zoomFactor * device.displayVideoZoomFactorMultiplier
  }
  return zoomFactor
}

class VideoCapture: NSObject, @unchecked Sendable {
  var predictor: Predictor!
  var previewLayer: AVCaptureVideoPreviewLayer?
  weak var delegate: VideoCaptureDelegate?
  var captureDevice: AVCaptureDevice?
  let captureSession = AVCaptureSession()
  var videoInput: AVCaptureDeviceInput? = nil
  let videoOutput = AVCaptureVideoDataOutput()
  let cameraQueue = DispatchQueue(label: "camera-queue")
  var inferenceOK = true
  var longSide: CGFloat = 3
  var shortSide: CGFloat = 4
  var frameSizeCaptured = false

  private var currentBuffer: CVPixelBuffer?
  // Called with the very next sample buffer rendered through the video output; matches upstream YOLO iOS
  // `captureNextFrame`. Used by `capturePhoto` so the share-sheet image is a freshly composited live frame (not a
  // separate AVCapturePhotoOutput still that would be off-axis from the preview).
  private var frameCaptureCompletion: ((UIImage?) -> Void)?
  // Monotonic token identifying the in-flight `captureNextFrame` request, so the watchdog only fires the completion it
  // armed (and not a newer one that replaced it).
  private var frameCaptureToken: UInt64 = 0
  private let imageContext = CIContext()

  func setUp(
    sessionPreset: AVCaptureSession.Preset = .hd1280x720,
    position: AVCaptureDevice.Position,
    videoOrientation: AVCaptureVideoOrientation,
    completion: @escaping (Bool) -> Void
  ) {
    // The Flutter plugin doesn't have a host UIViewController that can present a permission prompt, so we trigger the
    // iOS camera-permission dialog here on the first launch. Without this `setUpCamera` would bail with
    // `notDetermined` and the live preview would silently never come up. NSCameraUsageDescription must be set in the
    // host app's Info.plist (it is in the example).
    // Carry the non-Sendable completion across queue boundaries via SendableBox so Swift 6 strict concurrency doesn't
    // flag the captures. All invocations land on the main queue. `@Sendable` on `proceed` lets the closure cross into
    // `AVCaptureDevice.requestAccess`'s `@Sendable` callback below.
    let completionBox = SendableBox(completion)
    let proceed: @Sendable () -> Void = { [self] in
      cameraQueue.async {
        let success = self.setUpCamera(
          sessionPreset: sessionPreset, position: position, videoOrientation: videoOrientation)
        DispatchQueue.main.async {
          completionBox.value(success)
        }
      }
    }
    switch AVCaptureDevice.authorizationStatus(for: .video) {
    case .authorized:
      proceed()
    case .notDetermined:
      AVCaptureDevice.requestAccess(for: .video) { granted in
        if granted {
          proceed()
        } else {
          DispatchQueue.main.async { completionBox.value(false) }
        }
      }
    case .denied, .restricted:
      NSLog("YOLO VideoCapture: Camera permission denied or restricted. Cannot initialize camera.")
      DispatchQueue.main.async { completionBox.value(false) }
    @unknown default:
      DispatchQueue.main.async { completionBox.value(false) }
    }
  }

  func setUpCamera(
    sessionPreset: AVCaptureSession.Preset, position: AVCaptureDevice.Position,
    videoOrientation: AVCaptureVideoOrientation
  ) -> Bool {

    let authStatus = AVCaptureDevice.authorizationStatus(for: .video)
    if authStatus == .denied || authStatus == .restricted {
      NSLog("YOLO VideoCapture: Camera permission denied or restricted. Cannot initialize camera.")
      return false
    }

    if authStatus == .notDetermined {
      NSLog(
        "YOLO VideoCapture: Camera permission not determined. Please request permission first.")
      return false
    }

    captureSession.beginConfiguration()
    captureSession.sessionPreset = sessionPreset

    guard let device = bestCaptureDevice(position: position) else {
      NSLog(
        "YOLO VideoCapture: No camera device available for position: %@",
        String(describing: position))
      captureSession.commitConfiguration()
      return false
    }

    captureDevice = device

    let input: AVCaptureDeviceInput
    do {
      input = try AVCaptureDeviceInput(device: device)
    } catch {
      NSLog(
        "YOLO VideoCapture: Failed to create AVCaptureDeviceInput: %@", error.localizedDescription)
      captureSession.commitConfiguration()
      return false
    }

    videoInput = input

    if captureSession.canAddInput(input) {
      captureSession.addInput(input)
    } else {
      NSLog("YOLO VideoCapture: Cannot add video input to capture session")
      captureSession.commitConfiguration()
      return false
    }
    let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
    previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
    previewLayer.connection?.videoOrientation = videoOrientation
    self.previewLayer = previewLayer

    let settings: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
    ]

    videoOutput.videoSettings = settings
    videoOutput.alwaysDiscardsLateVideoFrames = true
    videoOutput.setSampleBufferDelegate(self, queue: cameraQueue)
    if captureSession.canAddOutput(videoOutput) {
      captureSession.addOutput(videoOutput)
    }

    let connection = videoOutput.connection(with: AVMediaType.video)
    connection?.videoOrientation = videoOrientation
    if position == .front {
      configureVideoMirroring(connection, isMirrored: true)
    }

    // Configure captureDevice
    guard let device = captureDevice else {
      NSLog("YOLO VideoCapture: captureDevice is nil, cannot configure")
      captureSession.commitConfiguration()
      return false
    }

    do {
      try device.lockForConfiguration()

      if device.isFocusModeSupported(AVCaptureDevice.FocusMode.continuousAutoFocus),
        device.isFocusPointOfInterestSupported
      {
        device.focusMode = AVCaptureDevice.FocusMode.continuousAutoFocus
        device.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
      }
      if device.isExposureModeSupported(.continuousAutoExposure) {
        device.exposureMode = .continuousAutoExposure
      }
      device.unlockForConfiguration()
    } catch {
      NSLog("YOLO VideoCapture: device configuration failed: %@", error.localizedDescription)
      captureSession.commitConfiguration()
      return false
    }

    captureSession.commitConfiguration()
    return true
  }

  func start() {
    if !captureSession.isRunning {
      DispatchQueue.global().async {
        self.captureSession.startRunning()
      }
    }
  }

  func stop() {
    // Drain any pending `captureNextFrame` completion on the cameraQueue (the same queue that stores it in
    // captureNextFrame and fires it in captureOutput). Without this, stopping before the next sample buffer arrives
    // strands the completion forever, hanging the Dart `await` behind pause/capturePhoto.
    cameraQueue.async { [weak self] in
      guard let self, let pending = self.frameCaptureCompletion else { return }
      self.frameCaptureCompletion = nil
      let completionBox = SendableBox(pending)
      DispatchQueue.main.async { completionBox.value(nil) }
    }
    if captureSession.isRunning {
      DispatchQueue.global().async {
        self.captureSession.stopRunning()
      }
    }
  }

  func setZoomRatio(ratio: CGFloat) {
    guard let device = captureDevice else {
      NSLog("YOLO VideoCapture: Cannot set zoom: captureDevice is nil")
      return
    }
    do {
      try device.lockForConfiguration()
      defer {
        device.unlockForConfiguration()
      }
      device.videoZoomFactor = ratio
    } catch {
      NSLog("YOLO VideoCapture: Failed to set zoom ratio: %@", error.localizedDescription)
    }
  }

  private func predictOnFrame(sampleBuffer: CMSampleBuffer) {
    guard let predictor = predictor else {
      return
    }
    if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      currentBuffer = pixelBuffer
      if !frameSizeCaptured {
        let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        longSide = max(frameWidth, frameHeight)
        shortSide = min(frameWidth, frameHeight)
        frameSizeCaptured = true
      }

      predictor.predict(sampleBuffer: sampleBuffer, onResultsListener: self, onInferenceTime: self)
      currentBuffer = nil
    }
  }

  func updateVideoOrientation(orientation: AVCaptureVideoOrientation) {
    guard let connection = videoOutput.connection(with: .video) else { return }

    connection.videoOrientation = orientation
    let currentInput = self.captureSession.inputs.first as? AVCaptureDeviceInput
    let isFront = currentInput?.device.position == .front
    configureVideoMirroring(connection, isMirrored: isFront)
    self.previewLayer?.connection?.videoOrientation = connection.videoOrientation
    configureVideoMirroring(self.previewLayer?.connection, isMirrored: isFront)
    frameSizeCaptured = false
  }

  /// Sets video mirroring deterministically — turn OFF automatic mirroring first so the explicit value sticks (iOS
  /// otherwise re-derives it). Mirrors yolo-ios-app VideoCapture.configureVideoMirroring.
  private func configureVideoMirroring(_ connection: AVCaptureConnection?, isMirrored: Bool) {
    guard let connection, connection.isVideoMirroringSupported else { return }
    connection.automaticallyAdjustsVideoMirroring = false
    connection.isVideoMirrored = isMirrored
  }

  deinit {
    if captureSession.isRunning {
      captureSession.stopRunning()
    }

    // Remove all inputs and outputs
    for input in captureSession.inputs {
      captureSession.removeInput(input)
    }
    for output in captureSession.outputs {
      captureSession.removeOutput(output)
    }
  }
}

extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
  /// Request a UIImage of the very next sample buffer rendered through the video output, matching upstream YOLO iOS
  /// `captureNextFrame`. Completion runs on the main queue. Returns `nil` if the session is stopped or a capture is
  /// already pending.
  func captureNextFrame(completion: @escaping (UIImage?) -> Void) {
    let completionBox = SendableBox(completion)
    cameraQueue.async { [weak self] in
      guard let self else { return }
      guard self.captureSession.isRunning, self.frameCaptureCompletion == nil else {
        DispatchQueue.main.async { completionBox.value(nil) }
        return
      }
      let pending = completionBox.value
      self.frameCaptureCompletion = pending
      self.frameCaptureToken &+= 1
      let token = self.frameCaptureToken
      // Defense-in-depth: if no sample buffer arrives within ~1s (e.g. the session is interrupted before the next
      // frame), fire the still-pending completion with nil so the caller's Dart `await` never hangs.
      self.cameraQueue.asyncAfter(deadline: .now() + 1.0) { [weak self] in
        guard let self, self.frameCaptureToken == token, let pending = self.frameCaptureCompletion
        else { return }
        self.frameCaptureCompletion = nil
        let completionBox = SendableBox(pending)
        DispatchQueue.main.async { completionBox.value(nil) }
      }
    }
  }

  func captureOutput(
    _ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    let pendingCompletion = frameCaptureCompletion
    if pendingCompletion != nil { frameCaptureCompletion = nil }
    defer {
      if let pendingCompletion {
        let image = CMSampleBufferGetImageBuffer(sampleBuffer).flatMap { pixelBuffer -> UIImage? in
          let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
          return imageContext.createCGImage(ciImage, from: ciImage.extent).map {
            UIImage(cgImage: $0)
          }
        }
        let imageBox = SendableBox(image)
        let completionBox = SendableBox(pendingCompletion)
        DispatchQueue.main.async { completionBox.value(imageBox.value) }
      }
    }
    guard inferenceOK else { return }
    predictOnFrame(sampleBuffer: sampleBuffer)
  }
}

extension VideoCapture: ResultsListener, InferenceTimeListener {
  func on(inferenceTime: Double, fpsRate: Double) {
    DispatchQueue.main.async {
      self.delegate?.onInferenceTime(speed: inferenceTime, fps: fpsRate)
    }
  }

  func on(result: YOLOResult) {
    DispatchQueue.main.async {
      self.delegate?.onPredict(result: result)
    }
  }
}
