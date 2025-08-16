cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

option(WITH_CUDA "Build with CUDA support" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
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

find_package(spdlog REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
# Find PipeWire using pkg-config
pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)
pkg_check_modules(SPA REQUIRED libspa-0.2)

include_directories(${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${PIPEWIRE_INCLUDE_DIRS}
        ${SPA_INCLUDE_DIRS}
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

# Create a shared library for the C++ detection functions
add_library(yolo_detection SHARED
           src/objectDetection2D.cc
           )


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