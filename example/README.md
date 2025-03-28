<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO Flutter Example App

This example application demonstrates how to integrate and utilize the [`ultralytics_yolo`](https://github.com/ultralytics/yolo-flutter-app) Flutter plugin to run powerful [Ultralytics YOLO](https://docs.ultralytics.com/) models directly within your Flutter projects. It serves as a practical starting point for developers looking to incorporate state-of-the-art [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, or other vision AI tasks into their mobile applications.

Explore the capabilities of running efficient [deep learning models](https://www.ultralytics.com/glossary/deep-learning-dl) on edge devices using Flutter and Ultralytics.

## 🚀 Getting Started

This project provides a basic implementation showcasing the core functionalities of the `ultralytics_yolo` plugin. If you're new to Flutter, these resources can help you get started:

- **[Lab: Write your first Flutter app](https://docs.flutter.dev/get-started/codelab):** A guided tutorial for beginners.
- **[Cookbook: Useful Flutter samples](https://docs.flutter.dev/cookbook):** Practical examples for common Flutter tasks.
- **[Flutter online documentation](https://docs.flutter.dev/):** Offers tutorials, samples, guidance on mobile development, and a full API reference.

### Prerequisites

- [Flutter SDK](https://docs.flutter.dev/get-started/install) installed on your machine.
- An editor like [VS Code](https://code.visualstudio.com/) with the Flutter plugin or [Android Studio](https://developer.android.com/studio).
- A physical device or emulator to run the app.

### Running the Example

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ultralytics/yolo-flutter-app.git
    cd yolo-flutter-app/example
    ```
2.  **Install dependencies:**
    ```bash
    flutter pub get
    ```
3.  **Run the app:**
    ```bash
    flutter run
    ```

This will launch the example application on your connected device or emulator. The app demonstrates loading an [Ultralytics YOLO model](https://docs.ultralytics.com/models/) (like [YOLOv8](https://docs.ultralytics.com/models/yolov8/) or [YOLO11](https://docs.ultralytics.com/models/yolo11/)) and performing inference on a sample image or live camera feed (depending on the example's implementation).

## ✨ Features Demonstrated

This example aims to illustrate:

- Initializing the `ultralytics_yolo` plugin.
- Loading a YOLO model (potentially exported to an edge-compatible format like [TFLite](https://docs.ultralytics.com/integrations/tflite/)).
- Running inference using the `predict` function. See the [Predict mode](https://docs.ultralytics.com/modes/predict/) documentation for more details.
- Processing and displaying the detection, segmentation, or pose estimation results. Learn about different [Ultralytics Tasks](https://docs.ultralytics.com/tasks/).
- Integrating with Flutter widgets for user interaction and display.

For more advanced use cases and deployment strategies, refer to the [Model Deployment Options guide](https://docs.ultralytics.com/guides/model-deployment-options/).

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this example application or the underlying plugin, please feel free to fork the repository, make your changes, and submit a pull request. Check out our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for more details on how to get involved with Ultralytics projects. We appreciate your support in making Vision AI accessible to everyone.
