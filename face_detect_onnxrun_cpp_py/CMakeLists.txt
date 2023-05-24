cmake_minimum_required(VERSION 3.22)
project(face_detect)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)

add_executable(face_detect main.cpp face_detector.cpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(face_detect PRIVATE "${ONNXRUNTIME_DIR}/include")
# link_directories("${ONNXRUNTIME_DIR}/lib")
target_compile_features(face_detect PRIVATE cxx_std_11)
target_link_libraries(face_detect ${OpenCV_LIBS})
target_link_libraries(face_detect "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")

