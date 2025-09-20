cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

option(WITH_CUDA "Build with CUDA support" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -fPIC")

if(WITH_CUDA)
    set(CUDA_ARCHS "52;61;72;75;86" CACHE STRING "List of architectures to generate device code for")
    project(cam_yolo_infer LANGUAGES C CXX CUDA)
endif()

project(cam_yolo_infer LANGUAGES C CXX) # C, C++

add_subdirectory(third_party/spdlog)
find_package(spdlog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

set(gRPC_INSTALL OFF CACHE BOOL "Disable gRPC installation")
set(protobuf_INSTALL OFF CACHE BOOL "Disable protobuf installation")
set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "Disable abseil installation")
set(utf8_range_ENABLE_INSTALL OFF CACHE BOOL "Disable utf8_range installation")

# Include gRPC as submodule
add_subdirectory(third_party/grpc ${CMAKE_CURRENT_BINARY_DIR}/grpc EXCLUDE_FROM_ALL)
message(STATUS "Using gRPC via add_subdirectory.")

# Set convenience variables (common.cmake GRPC_AS_SUBMODULE case)
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_ORCA_SERVICE grpcpp_orca_service)
set(_GRPC_GRPCPP grpc++)

# Compiler executables (as common.cmake GRPC_AS_SUBMODULE case)
if(CMAKE_CROSSCOMPILING)
    find_program(_PROTOBUF_PROTOC protoc)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
    set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()

# Find PipeWire using pkg-config
pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)
pkg_check_modules(SPA REQUIRED libspa-0.2)

set(CAMERA_IDL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/camera_idl")
set(coordinator_proto_srcs "${CAMERA_IDL_DIR}/gen/coordinator.pb.cc")
set(coordinator_proto_hdrs "${CAMERA_IDL_DIR}/gen/coordinator.pb.h")
set(coordinator_grpc_srcs "${CAMERA_IDL_DIR}/gen/coordinator.grpc.pb.cc")
set(coordinator_grpc_hdrs "${CAMERA_IDL_DIR}/gen/coordinator.grpc.pb.h")

# Custom target to generate C++ proto files
add_custom_command(
        OUTPUT "${coordinator_proto_srcs}" "${coordinator_proto_hdrs}" "${coordinator_grpc_srcs}" "${coordinator_grpc_hdrs}"
        COMMAND ${CAMERA_IDL_DIR}/build.sh cpp
        WORKING_DIRECTORY ${CAMERA_IDL_DIR}
        COMMENT "Generating C++ protobuf files from camera_idl"
)

include_directories(${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${PIPEWIRE_INCLUDE_DIRS}
        ${SPA_INCLUDE_DIRS}
        ${CAMERA_IDL_DIR}/gen  # generated proto files
)

if(WITH_CUDA)
    enable_language(CUDA)
    message("${CUDA_LIBRARIES}")
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DUSE_CUDA)

    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

link_directories(${OpenCV_LIBRARY_DIRS}
                 ${PIPEWIRE_LIBRARY_DIRS}
                )

# gRPC proto library
add_library(coordinator_grpc_proto STATIC
        ${coordinator_grpc_srcs}
        ${coordinator_grpc_hdrs}
        ${coordinator_proto_srcs}
        ${coordinator_proto_hdrs})

target_compile_options(coordinator_grpc_proto PRIVATE -fPIC)

target_link_libraries(coordinator_grpc_proto
        absl::check
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})

# Create a shared library for the C++ detection functions
add_library(yolo_detection SHARED
           src/objectDetection2D.cc
           src/grpc_client.cc
           )

target_compile_options(yolo_detection PRIVATE -fPIC)

target_link_libraries(yolo_detection
        coordinator_grpc_proto
        ${OpenCV_LIBRARIES}
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF}
        spdlog::spdlog
)

if(WITH_CUDA)
    target_link_libraries(yolo_detection CUDA::cudart)
endif ()

# filter_g program
add_executable(filter_g
    src/filter_g.c
)

target_include_directories(filter_g
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${PIPEWIRE_INCLUDE_DIRS}
    ${SPA_INCLUDE_DIRS}
)

target_link_libraries(filter_g
    yolo_detection
    stdc++ 
    ${PIPEWIRE_LIBRARIES}
    ${OpenCV_LIBRARIES}
    spdlog::spdlog
)

if(WITH_CUDA)
    target_link_libraries(filter_g CUDA::cudart)
endif ()

# Set linker language to CXX for proper C++ runtime linking
set_target_properties(filter_g PROPERTIES
    LINKER_LANGUAGE CXX
)