CMAKE_MINIMUM_REQUIRED(VERSION 3.0.0)
project(YOLOv8CPP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.8.0 REQUIRED)

set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/onnxruntime-linux-x64-gpu-1.17.0/")

set(ONNXRUNTIME_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime-linux-x64-gpu-1.17.0/libonnxruntime.so")
set(ONNXRUNTIME_PROVIDERS_SHARED_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime-linux-x64-gpu-1.17.0/libonnxruntime_providers_shared.so")
set(ONNXRUNTIME_PROVIDERS_CUDA_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime-linux-x64-gpu-1.17.0/libonnxruntime_providers_cuda.so")

include_directories(include ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIR})

file(GLOB_RECURSE CURR_SOURCES src/*.cpp)

add_executable(YOLOv8CPP ${CURR_SOURCES})
target_include_directories(YOLOv8CPP PRIVATE "include")

target_link_libraries(YOLOv8CPP
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
    ${ONNXRUNTIME_PROVIDERS_SHARED_LIBRARY}
    ${ONNXRUNTIME_PROVIDERS_CUDA_LIBRARY}
)