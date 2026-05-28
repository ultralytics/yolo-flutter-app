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
  if UserDefaults.standard.bool(forKey: "use_telephoto"),
    let device = AVCaptureDevice.default(.builtInTelephotoCamera, for: .video, position: position)
  {
    return device
  } else if let device = AVCaptureDevice.default(
    .builtInDualCamera, for: .video, position: position)
  {
    return device
  } else if let device = AVCaptureDevice.default(
    .builtInWideAngleCamera, for: .video, position: position)
  {
    return device
  } else {
    return nil
  }
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
  private let imageContext = CIContext()

  func setUp(
    sessionPreset: AVCaptureSession.Preset = .hd1280x720,
    position: AVCaptureDevice.Position,
    videoOrientation: AVCaptureVideoOrientation,
    completion: @escaping (Bool) -> Void
  ) {
    cameraQueue.async {
      let success = self.setUpCamera(
        sessionPreset: sessionPreset, position: position, videoOrientation: videoOrientation)
      DispatchQueue.main.async {
        completion(success)
      }
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
      connection?.isVideoMirrored = true
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
      device.exposureMode = AVCaptureDevice.ExposureMode.continuousAutoExposure
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
    if currentInput?.device.position == .front {
      connection.isVideoMirrored = true
    } else {
      connection.isVideoMirrored = false
    }
    self.previewLayer?.connection?.videoOrientation = connection.videoOrientation
    frameSizeCaptured = false
  }

  deinit {
    if captureSession.isRunning {
      captureSession.stopRunning()
    }

    // Remove all inputs and outputs
    if let inputs = captureSession.inputs as? [AVCaptureInput] {
      for input in inputs {
        captureSession.removeInput(input)
      }
    }

    if let outputs = captureSession.outputs as? [AVCaptureOutput] {
      for output in outputs {
        captureSession.removeOutput(output)
      }
    }
  }
}

extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
  /// Request a UIImage of the very next sample buffer rendered through the video output, matching upstream YOLO iOS
  /// `captureNextFrame`. Completion runs on the main queue. Returns `nil` if the session is stopped or a capture is
  /// already pending.
  func captureNextFrame(completion: @escaping (UIImage?) -> Void) {
    cameraQueue.async { [weak self] in
      guard let self else { return }
      guard self.captureSession.isRunning, self.frameCaptureCompletion == nil else {
        DispatchQueue.main.async { completion(nil) }
        return
      }
      self.frameCaptureCompletion = completion
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
        DispatchQueue.main.async { pendingCompletion(image) }
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
