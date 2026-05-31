# Keep SnakeYAML classes
-keep class org.yaml.snakeyaml.** { *; }
-dontwarn org.yaml.snakeyaml.**

# Keep java.beans classes for SnakeYAML
-keep class java.beans.** { *; }
-dontwarn java.beans.**

# Keep property utilities
-keep class org.yaml.snakeyaml.introspector.** { *; }
-keep class org.yaml.snakeyaml.constructor.** { *; }
-keep class org.yaml.snakeyaml.representer.** { *; }

# Keep TensorFlow Lite classes
-keep class org.tensorflow.** { *; }
-keep interface org.tensorflow.** { *; }

# Keep LiteRT 2.x ("LiteRT Next") classes — CompiledModel/Accelerator/TensorBuffer/Environment are called from the
# native libLiteRt.so via JNI/reflection, so R8 must not strip or rename them (otherwise release builds crash in
# ObjectDetector.<init> while creating the CompiledModel).
-keep class com.google.ai.edge.litert.** { *; }
-keep interface com.google.ai.edge.litert.** { *; }
-dontwarn com.google.ai.edge.litert.**

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}