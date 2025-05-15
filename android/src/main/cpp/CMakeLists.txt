cmake_minimum_required(VERSION 3.10)

project("ultralytics")

# Note OpenCV removed from repository in https://github.com/ultralytics/yolo-flutter-app/pull/127
# set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.6.0-android/sdk/native/jni)
# find_package(OpenCV REQUIRED core imgproc)

add_library(${CMAKE_PROJECT_NAME} SHARED
        tflite_detect.cpp
        tflite_segment.cpp)

find_library(
        log-lib
        log)

target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        ${log-lib}
        segment
        omp
        # ${OpenCV_LIBS}
        )
