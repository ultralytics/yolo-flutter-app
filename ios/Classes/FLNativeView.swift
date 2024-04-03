import Flutter
import UIKit
import AVFoundation
// import Ultralytics

class FLNativeView: NSObject, FlutterPlatformView{
    private let previewView: UIView
    private let videoCapture: VideoCapture
    private var busy = false
    
    init(
        frame: CGRect,
        viewIdentifier viewId: Int64,
        arguments args: Any?,
        videoCapture: VideoCapture
    ) {
        let screenSize: CGRect = UIScreen.main.bounds
        previewView = UIView(frame: CGRect(x: 0, y: 0, width: screenSize.width, height: screenSize.height))
        
        self.videoCapture = videoCapture
        
        super.init()
        
        startCameraPreview(position: .back)
    }
    
    func view() -> UIView {
        return previewView
    }
    
    private func startCameraPreview(position: AVCaptureDevice.Position){
        if !busy {
            busy = true
            
            videoCapture.setUp(sessionPreset: .photo, position: position) { success in
                // .hd4K3840x2160 or .photo (4032x3024)  Warning: 4k may not work on all devices i.e. 2019 iPod
                if success {
                    // Add the video preview into the UI.
                    if let previewLayer = self.videoCapture.previewLayer {
                        self.previewView.layer.addSublayer(previewLayer)
                        self.videoCapture.previewLayer?.frame = self.previewView.bounds  // resize preview layer
                    }
                    
                    // Once everything is set up, we can start capturing live video.
                    self.videoCapture.start()
                    
                    self.busy = false
                }
            }
        }
    }
    
    
}

