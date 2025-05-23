#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint yolo.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'ultralytics_yolo'
  s.version          = '0.0.4'
  s.summary          = 'Flutter plugin for YOLO (You Only Look Once) models'
  s.description      = <<-DESC
Flutter plugin for YOLO (You Only Look Once) models, supporting object detection, segmentation, classification, pose estimation and oriented bounding boxes (OBB) on both Android and iOS.
                       DESC
  s.homepage         = 'https://github.com/ultralytics/yolo-flutter-app'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Ultralytics' => 'info@ultralytics.com' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '13.0'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'

  # If your plugin requires a privacy manifest, for example if it uses any
  # required reason APIs, update the PrivacyInfo.xcprivacy file to describe your
  # plugin's privacy impact, and then uncomment this line. For more information,
  # see https://developer.apple.com/documentation/bundleresources/privacy_manifest_files
  s.resource_bundles = {'ultralytics_yolo_privacy' => ['Resources/PrivacyInfo.xcprivacy']}
end
