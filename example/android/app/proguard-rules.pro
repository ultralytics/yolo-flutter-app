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

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}