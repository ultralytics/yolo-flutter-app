import AVFoundation
import CoreVideo
import UIKit

public protocol VideoCaptureDelegate: AnyObject {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame: CMSampleBuffer)
}

func bestCaptureDevice(position: AVCaptureDevice.Position) -> AVCaptureDevice {
    // print("USE TELEPHOTO: ")
    // print(UserDefaults.standard.bool(forKey: "use_telephoto"))

    if UserDefaults.standard.bool(forKey: "use_telephoto"), let device = AVCaptureDevice.default(.builtInTelephotoCamera, for: .video, position: position) {
        return device
    } else if let device = AVCaptureDevice.default(.builtInDualCamera, for: .video, position: position) {
        return device
    } else if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) {
        return device
    } else {
        fatalError("Missing expected back camera device.")
    }
}

public class VideoCapture: NSObject {
    public var previewLayer: AVCaptureVideoPreviewLayer?
    public weak var delegate: VideoCaptureDelegate?
    var captureDevice: AVCaptureDevice?
    let captureSession = AVCaptureSession()
    var videoInput: AVCaptureDeviceInput? = nil
    let videoOutput = AVCaptureVideoDataOutput()
    var photoOutput = AVCapturePhotoOutput()
    let cameraQueue = DispatchQueue(label: "camera-queue")
    var lastCapturedPhoto: UIImage? = nil

    public func setUp(sessionPreset: AVCaptureSession.Preset = .hd1280x720,
                      position: AVCaptureDevice.Position,
                      completion: @escaping (Bool) -> Void) {
        cameraQueue.async {
            let success = self.setUpCamera(sessionPreset: sessionPreset, position: position)
            DispatchQueue.main.async {
                completion(success)
            }
        }
    }

    func setUpCamera(sessionPreset: AVCaptureSession.Preset, position: AVCaptureDevice.Position) -> Bool {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = sessionPreset

        captureDevice = bestCaptureDevice(position: position)
        videoInput = try! AVCaptureDeviceInput(device: captureDevice!)

        if captureSession.canAddInput(videoInput!) {
            captureSession.addInput(videoInput!)
        }

        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        previewLayer.connection?.videoOrientation = .portrait
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
        if captureSession.canAddOutput(photoOutput) {
            captureSession.addOutput(photoOutput)
            photoOutput.isHighResolutionCaptureEnabled = true
            photoOutput.isLivePhotoCaptureEnabled = photoOutput.isLivePhotoCaptureSupported
        }

        // We want the buffers to be in portrait orientation otherwise they are
        // rotated by 90 degrees. Need to set this _after_ addOutput()!
        // let curDeviceOrientation = UIDevice.current.orientation
        let connection = videoOutput.connection(with: AVMediaType.video)
        connection?.videoOrientation = .portrait
        if position == .front{
            connection?.isVideoMirrored = true
        }

        // Configure captureDevice
        do {
            try captureDevice!.lockForConfiguration()
        } catch {
            print("device configuration not working")
        }
        // captureDevice.setFocusModeLocked(lensPosition: 1.0, completionHandler: { (time) -> Void in })
        if captureDevice!.isFocusModeSupported(AVCaptureDevice.FocusMode.continuousAutoFocus), captureDevice!.isFocusPointOfInterestSupported {
            captureDevice!.focusMode = AVCaptureDevice.FocusMode.continuousAutoFocus
            captureDevice!.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
        }
        captureDevice!.exposureMode = AVCaptureDevice.ExposureMode.continuousAutoExposure
        captureDevice!.unlockForConfiguration()

        captureSession.commitConfiguration()
        return true
    }

    public func start() {
        if !captureSession.isRunning {
            DispatchQueue.global().async {
                self.captureSession.startRunning()
            }
        }
    }

    public func stop() {
        if captureSession.isRunning {
            DispatchQueue.global().async {
                self.captureSession.stopRunning()
            }
        }
    }
    
    public func setZoomRatio(ratio: CGFloat){
        do {
            try captureDevice!.lockForConfiguration()
            defer {
                captureDevice!.unlockForConfiguration()
            }
            captureDevice!.videoZoomFactor = ratio
        } catch { }
    }
}

extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        delegate?.videoCapture(self, didCaptureVideoFrame: sampleBuffer)
    }
}

extension VideoCapture: AVCapturePhotoCaptureDelegate {
    @available(iOS 11.0, *)
    public func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard let data = photo.fileDataRepresentation(),
              let image =  UIImage(data: data) else {
                return
        }

        self.lastCapturedPhoto = image
    }
}
