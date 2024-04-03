cmake_minimum_required(VERSION 3.10)

project("ultralytics")

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.6.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

add_library(${CMAKE_PROJECT_NAME} SHARED
        tflite_detect.cpp)

find_library(
        log-lib
        log)

target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        ${log-lib}
        ${OpenCV_LIBS}
        )