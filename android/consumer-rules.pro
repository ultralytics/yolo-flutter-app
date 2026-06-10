# Consumer ProGuard/R8 rules shipped with the ultralytics_yolo plugin. Applied automatically to any release build of an
# app that depends on the plugin, so they don't have to know about these internals.

# LiteRT 2.x ("LiteRT Next"): CompiledModel / Accelerator / TensorBuffer / Environment are invoked from the native
# libLiteRt.so via JNI/reflection. R8 must not strip or rename them, or release builds crash in the predictor's
# <init> when creating the CompiledModel.
-keep class com.google.ai.edge.litert.** { *; }
-keep interface com.google.ai.edge.litert.** { *; }
-dontwarn com.google.ai.edge.litert.**

# Legacy TFLite / LiteRT metadata (MetadataExtractor reads embedded model metadata via flatbuffers + reflection).
-keep class org.tensorflow.** { *; }
-keep interface org.tensorflow.** { *; }
-dontwarn org.tensorflow.**

# SnakeYAML parses model metadata.yaml and uses reflection over its own classes.
-keep class org.yaml.snakeyaml.** { *; }
-dontwarn org.yaml.snakeyaml.**

# JNI entry points.
-keepclasseswithmembernames class * {
    native <methods>;
}

# ONNX Runtime QNN is an optional (compileOnly) dependency; apps that don't add it must not fail R8 shrinking.
-dontwarn ai.onnxruntime.**
