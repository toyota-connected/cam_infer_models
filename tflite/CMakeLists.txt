cmake_minimum_required(VERSION 3.17)
project(cam_tflite_infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fPIC")

# find_package(spdlog REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)
pkg_check_modules(SPA REQUIRED libspa-0.2)

add_subdirectory(../third_party/spdlog spdlog_build)
add_subdirectory(../third_party/abseil-cpp abseil_build)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tensorflow/
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tensorflow/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/
    ${PIPEWIRE_INCLUDE_DIRS}
    ${SPA_INCLUDE_DIRS}
)

link_directories(
    ${OpenCV_LIBRARY_DIRS}
    ${PIPEWIRE_LIBRARY_DIRS}
)

add_library(tensorflow-lite SHARED IMPORTED)
set_property(TARGET tensorflow-lite PROPERTY IMPORTED_LOCATION
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/tensorflow/tensorflow/lite/build/libtensorflow-lite.so
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(tflite_detection SHARED objectDetection_tflite.cc)
target_link_libraries(tflite_detection
    PRIVATE
    tensorflow-lite
    spdlog::spdlog
    absl::log absl::base absl::synchronization absl::strings
)

add_executable(${PROJECT_NAME} ../src/filter_g.c)
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/
    ${PIPEWIRE_INCLUDE_DIRS}
    ${SPA_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    tflite_detection
    stdc++
    ${PIPEWIRE_LIBRARIES}
    ${OpenCV_LIBRARIES}
    spdlog::spdlog
    absl::log absl::base absl::synchronization absl::strings
    tensorflow-lite
    pthread
)

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

