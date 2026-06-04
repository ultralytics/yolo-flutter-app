#
# To learn more about a Podspec see https://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint ultralytics_yolo.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'ultralytics_yolo'
  s.version          = '0.5.1'
  s.summary          = 'Flutter plugin for YOLO (You Only Look Once) models'
  s.description      = <<-DESC
Flutter plugin for YOLO (You Only Look Once) models, supporting object detection, segmentation, classification, pose estimation and oriented bounding boxes (OBB) on both Android and iOS.
                       DESC
  s.homepage         = 'https://github.com/ultralytics/yolo-flutter-app'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Ultralytics' => 'info@ultralytics.com' }
  s.source           = { :path => '.' }
  # Restrict to Swift/Obj-C sources — globbing `**/*` was sweeping in `Classes/README.md` and triggering a
  # `no rule to process file ... net.daringfireball.markdown` warning on every Xcode build.
  s.source_files = 'Classes/**/*.{swift,h,m}'
  s.dependency 'Flutter'
  s.dependency 'UltralyticsYOLO', '>= 8.9.2', '< 9.0'
  s.platform = :ios, '13.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'

  # If your plugin requires a privacy manifest, for example if it uses any
  # required reason APIs, update the PrivacyInfo.xcprivacy file to describe your
  # plugin's privacy impact, and then uncomment this line. For more information,
  # see https://developer.apple.com/documentation/bundleresources/privacy-manifest-files
  s.resource_bundles = {'ultralytics_yolo_privacy' => ['Resources/PrivacyInfo.xcprivacy']}
end
