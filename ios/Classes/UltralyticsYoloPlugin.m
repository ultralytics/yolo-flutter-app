#import "UltralyticsYoloPlugin.h"
#if __has_include(<ultralytics_yolo/ultralytics_yolo-Swift.h>)
#import <ultralytics_yolo/ultralytics_yolo-Swift.h>
#else
// Support project import fallback if the generated compatibility header
// is not copied when this plugin is created as a library.
// https://forums.swift.org/t/swift-static-libraries-dont-copy-generated-objective-c-header/19816
#import "ultralytics_yolo-Swift.h"
#endif

@implementation UltralyticsYoloPlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftUltralyticsYoloPlugin registerWithRegistrar:registrar];
}
@end
