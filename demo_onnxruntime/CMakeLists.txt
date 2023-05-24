cmake_minimum_required(VERSION 3.22)
project(test)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)

add_executable(test test.cpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(test PRIVATE "${ONNXRUNTIME_DIR}/include")
target_compile_features(test PRIVATE cxx_std_11)
target_link_libraries(test ${OpenCV_LIBS})
target_link_libraries(test "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")

