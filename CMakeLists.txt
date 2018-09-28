cmake_minimum_required(VERSION 2.8.3)
project(parallax)

# default to release build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)
set(CMAKE_CXX_FLAGS "-std=c++14 -mavx -openmp") #-O2 -mfma

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(Ceres REQUIRED)

include_directories(include
  # lib/common_cpp/include
  ${EIGEN3_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  # ${OPENCV_INCLUDE_DIRS}
  # ${CERES_INCLUDE_DIRS}
)

add_executable(test_rcpo
  src/test.cpp
  src/ptr/GN_step.cpp
)
target_link_libraries(test_rcpo ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} stdc++fs)