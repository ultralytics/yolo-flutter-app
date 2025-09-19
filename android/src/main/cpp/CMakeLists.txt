cmake_minimum_required(VERSION 3.10.2)
project("ultralytics")

add_library(
ultralytics           # ライブラリ名（後述の System.loadLibrary で指定）
SHARED
native-lib.cpp        # C++ ソースファイル
)

# NDK r28 automatically handles 16KB page size for arm64-v8a and x86_64
# The following configuration is kept for explicit documentation
# but is no longer strictly necessary with NDK r28
if(ANDROID_ABI STREQUAL "arm64-v8a" OR ANDROID_ABI STREQUAL "x86_64")
    message(STATUS "NDK r28: 16KB page size support enabled by default for ${ANDROID_ABI}")
endif()

find_library(
log-lib
log
)

target_link_libraries(
ultralytics
${log-lib}
)

