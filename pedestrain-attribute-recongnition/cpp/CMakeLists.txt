cmake_minimum_required(VERSION 3.22)
project(pede_recog)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)

add_executable(pede_recog main.cpp pede_attr1.cpp pede_attr1.hpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(pede_recog PRIVATE "${ONNXRUNTIME_DIR}/include")
# link_directories("${ONNXRUNTIME_DIR}/lib")
target_compile_features(pede_recog PRIVATE cxx_std_11)
target_link_libraries(pede_recog ${OpenCV_LIBS})
target_link_libraries(pede_recog "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")

