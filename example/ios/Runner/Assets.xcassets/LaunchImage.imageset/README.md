<a href="https://www.ultralytics.com/"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# iOS Launch Screen Assets

This directory contains the image assets used for the launch screen of the iOS version of your Flutter application.

## 🖼️ Customizing the Launch Screen

You have two primary methods to customize the launch screen with your own assets:

1.  **Direct File Replacement:** Replace the existing image files within this `LaunchImage.imageset` directory with your desired images. Ensure your new images match the required names and dimensions expected by iOS. You can find more details in the [Apple Human Interface Guidelines for Launch Screens](https://developer.apple.com/design/human-interface-guidelines/launch-screens).

2.  **Using Xcode:**
    - Open your Flutter project's iOS workspace by running `open ios/Runner.xcworkspace` in your terminal from the project root.
    - In Xcode, navigate to `Runner/Assets.xcassets` in the Project Navigator on the left sidebar.
    - Select the `LaunchImage` asset catalog.
    - Drag and drop your new image files into the appropriate placeholders within the Xcode interface. Xcode helps manage different resolutions and device types. For more guidance on using asset catalogs, refer to the [Xcode documentation](https://developer.apple.com/documentation/xcode/managing-assets-with-asset-catalogs).

Choose the method that best suits your workflow. Using Xcode is generally recommended as it provides a visual interface and helps manage different device requirements effectively. For general Flutter development practices, check out the official [Flutter documentation](https://docs.flutter.dev/).

---

We hope this guide helps you customize your application's launch screen! Contributions to improve this documentation or the example app are welcome. Please see the main repository's contribution guidelines for more information.
