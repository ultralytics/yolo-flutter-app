// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  BoundingBoxView for Ultralytics YOLO App
//  This class is designed to visualize bounding boxes and labels for detected objects in the YOLO models within the Ultralytics YOLO app.
//  It leverages Core Animation layers to draw the bounding boxes and text labels dynamically on the detection video feed.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  BoundingBoxView facilitates the clear representation of detection results, improving user interaction with the app by
//  providing immediate visual feedback on detected objects, including their classification and confidence level.

import Foundation
import UIKit
import UltralyticsYOLO

/// Manages the visualization of bounding boxes and associated labels for object detection results.
@MainActor
class BoundingBoxView {
  /// The layer that draws the bounding box around a detected object.
  let shapeLayer: CAShapeLayer

  /// The layer that displays the label and confidence score for the detected object.
  let textLayer: CATextLayer

  /// The base font size that can be scaled for external displays.
  private var baseFontSize: CGFloat = 11

  /// Initializes a new BoundingBoxView with configured shape and text layers.
  init() {
    shapeLayer = CAShapeLayer()
    shapeLayer.fillColor = UIColor.clear.cgColor  // No fill to only show the bounding outline
    shapeLayer.lineWidth = 4  // Set the stroke line width
    shapeLayer.isHidden = true  // Initially hidden; shown when a detection occurs

    textLayer = CATextLayer()
    textLayer.isHidden = true  // Initially hidden; shown with label when a detection occurs
    DetectionLabelStyle.configure(textLayer, fontSize: baseFontSize)
  }

  /// Sets the font size for the text layer (useful for external displays).
  func setFontSize(_ size: CGFloat) {
    baseFontSize = size
    DetectionLabelStyle.configure(textLayer, fontSize: size)
  }

  /// Sets the line width for the bounding box (useful for external displays).
  func setLineWidth(_ width: CGFloat) {
    shapeLayer.lineWidth = width
  }

  /// Adds the bounding box and text layers to a specified parent layer.
  /// - Parameter parent: The CALayer to which the bounding box and text layers will be added.
  func addToLayer(_ parent: CALayer) {
    parent.addSublayer(shapeLayer)
    parent.addSublayer(textLayer)
  }

  /// Updates the bounding box and label to be visible with specified properties.
  /// - Parameters:
  ///   - frame: The CGRect frame defining the bounding box's size and position.
  ///   - label: The text label to display (e.g., object class and confidence).
  ///   - color: The class color for the box stroke and label background, and its label text color.
  ///   - alpha: The opacity level for the bounding box stroke and label.
  ///   - angle: Optional rotation angle in radians for oriented boxes.
  func show(frame: CGRect, label: String, color: ClassColor, alpha: CGFloat, angle: CGFloat? = nil) {
    CATransaction.begin()
    CATransaction.setDisableActions(true)  // Disable implicit animations

    let path = UIBezierPath(roundedRect: frame, cornerRadius: 6.0)
    if let angle {
      var transform = CGAffineTransform(translationX: frame.midX, y: frame.midY)
      transform = transform.rotated(by: angle)
      transform = transform.translatedBy(x: -frame.midX, y: -frame.midY)
      path.apply(transform)
    }
    shapeLayer.path = path.cgPath
    // Opaque class colors with alpha on the layers, so no colors are allocated per box per frame.
    shapeLayer.strokeColor = color.color.cgColor
    shapeLayer.opacity = Float(alpha)
    shapeLayer.isHidden = false

    textLayer.string = label
    textLayer.backgroundColor = color.color.cgColor
    textLayer.foregroundColor = color.textColor.cgColor
    textLayer.opacity = Float(alpha)

    // Place the label with the shared style, clamped into the parent layer; hide it when the box is off-screen.
    let bounds = shapeLayer.superlayer?.bounds ?? .zero
    textLayer.isHidden = !DetectionLabelStyle.isVisible(path.bounds, within: bounds)
    if !textLayer.isHidden {
      textLayer.frame = DetectionLabelStyle.frame(
        for: label, fontSize: textLayer.fontSize, anchor: angle == nil ? frame.origin : path.bounds.origin,
        within: bounds)
    }
    CATransaction.commit()
  }

  /// Hides the bounding box and text layers.
  func hide() {
    shapeLayer.isHidden = true
    textLayer.isHidden = true
  }
}
