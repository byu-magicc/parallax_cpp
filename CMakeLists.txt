cmake_minimum_required(VERSION 2.8.3)
project(parallax)

# default to release build
# Note: Release mode disables assert statements, so be wary when testing.
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-std=c++14 -mavx -openmp") #-O2 -mfma

# Option to categorize times in detail
if(TIME_VERBOSE)
	add_definitions(-DTIME_VERBOSE=1)
endif()

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(Ceres REQUIRED)

include_directories(
	lib/5_point/include
	lib/common/include
	lib/gnsac_ptr_eigen/include
	lib/gnsac_ptr_opencv/include
	# lib/common_cpp/include
	${EIGEN3_INCLUDE_DIRS}
	${YAML_CPP_INCLUDE_DIR}
	# ${OPENCV_INCLUDE_DIRS}
	# ${CERES_INCLUDE_DIRS}
)

add_executable(cli
	src/cli.cpp
	lib/common/src/common.cpp
	lib/common/src/comm_loaders.cpp
	lib/common/src/comm_math.cpp
	lib/common/src/solvers.cpp
	lib/gnsac_ptr_eigen/src/gnsac_ptr_eig.cpp
	lib/gnsac_ptr_opencv/src/gnsac_ptr_ocv.cpp
)

add_executable(det_test_ocv
	lib/gnsac_ptr_opencv/src/determinism_test.cpp
	lib/gnsac_ptr_opencv/src/gnsac_ptr_ocv.cpp
	lib/common/src/common.cpp
	lib/common/src/comm_loaders.cpp
	lib/common/src/comm_math.cpp
	lib/common/src/solvers.cpp
)

add_executable(det_test_eig
	lib/gnsac_ptr_eigen/src/determinism_test.cpp
	lib/gnsac_ptr_eigen/src/gnsac_ptr_eig.cpp
	lib/common/src/common.cpp
	lib/common/src/comm_loaders.cpp
	lib/common/src/comm_math.cpp
	lib/common/src/solvers.cpp
)

target_link_libraries(cli ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} stdc++fs)
target_link_libraries(det_test_ocv ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} stdc++fs)
target_link_libraries(det_test_eig ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} stdc++fs)