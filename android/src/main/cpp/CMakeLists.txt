cmake_minimum_required(VERSION 3.10.2)
project("ultralytics")

add_library(
ultralytics           # ライブラリ名（後述の System.loadLibrary で指定）
SHARED
native-lib.cpp        # C++ ソースファイル
)

find_library(
log-lib
log
)

target_link_libraries(
ultralytics
${log-lib}
)

