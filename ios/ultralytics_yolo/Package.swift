// swift-tools-version: 5.9
// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// Swift Package Manager manifest for the ultralytics_yolo Flutter plugin's iOS layer. CocoaPods (see
// ../ultralytics_yolo.podspec) and SwiftPM compile the same Sources/ultralytics_yolo tree; keep the two in sync.

import PackageDescription

let package = Package(
  name: "ultralytics_yolo",
  platforms: [
    .iOS(.v13)
  ],
  products: [
    // Flutter requires the library name to be the dash-separated form of the underscore target name.
    .library(name: "ultralytics-yolo", targets: ["ultralytics_yolo"])
  ],
  dependencies: [
    .package(name: "FlutterFramework", path: "../FlutterFramework"),
    // Shared YOLO inference core, the single source of truth for iOS (https://github.com/ultralytics/yolo-ios-app).
    // Mirrors the CocoaPods `s.dependency 'UltralyticsYOLO', '>= 8.9.17', '< 9.0'` range in the podspec.
    .package(url: "https://github.com/ultralytics/yolo-ios-app.git", "8.9.17"..<"9.0.0"),
  ],
  targets: [
    .target(
      name: "ultralytics_yolo",
      dependencies: [
        .product(name: "FlutterFramework", package: "FlutterFramework"),
        .product(name: "UltralyticsYOLO", package: "yolo-ios-app"),
      ],
      exclude: ["README.md"],
      resources: [
        .process("PrivacyInfo.xcprivacy")
      ]
    )
  ]
)
