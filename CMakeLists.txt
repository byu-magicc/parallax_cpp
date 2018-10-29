cmake_minimum_required(VERSION 2.8.3)
project(gnsac)

# default to release build
# Note: Release mode disables assert statements, so be wary when testing.
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-std=c++14 -mavx -openmp") #-O2 -mfma

# Option to categorize times in detail
option(TIME_VERBOSE "Option to categorize times in detail" OFF)
if(TIME_VERBOSE)
	add_definitions(-DTIME_VERBOSE)
endif()

# Find libraries
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(Ceres REQUIRED)

include_directories(
	lib/five_point_opencv/include
	lib/common/include
	lib/gnsac_ptr_eigen/include
	lib/gnsac_ptr_opencv/include
	lib/gnsac_eigen/include
	${EIGEN3_INCLUDE_DIRS}
	${YAML_CPP_INCLUDE_DIR}
	${OPENCV_INCLUDE_DIRS}
	# ${CERES_INCLUDE_DIRS}
)

add_executable(cli
	src/cli.cpp
	lib/common/src/common.cpp
	lib/common/src/comm_loaders.cpp
	lib/common/src/comm_math.cpp
	lib/common/src/solvers.cpp
	lib/five_point_opencv/src/five-point.cpp
	lib/five_point_opencv/src/levmarq.cpp
	lib/five_point_opencv/src/preempt.cpp
	lib/five_point_opencv/src/ptsetreg.cpp
	lib/gnsac_ptr_eigen/src/gnsac_ptr_eig.cpp
	lib/gnsac_ptr_opencv/src/gnsac_ptr_ocv.cpp
	lib/gnsac_eigen/src/gnsac_eigen.cpp
)
target_link_libraries(cli ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} stdc++fs)

# add_executable(det_test_ocv
# 	lib/gnsac_ptr_opencv/src/determinism_test.cpp
# 	lib/gnsac_ptr_eigen/src/gnsac_ptr_eig.cpp
# 	lib/gnsac_ptr_opencv/src/gnsac_ptr_ocv.cpp
# 	lib/common/src/common.cpp
# 	lib/common/src/comm_loaders.cpp
# 	lib/common/src/comm_math.cpp
# 	lib/common/src/solvers.cpp
# 	lib/gnsac_eigen/src/gnsac_eigen.cpp
# )
# target_link_libraries(det_test_ocv ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} stdc++fs)

# add_executable(det_test_eig
# 	lib/gnsac_ptr_eigen/src/determinism_test.cpp
# 	lib/gnsac_ptr_eigen/src/gnsac_ptr_eig.cpp
# 	lib/gnsac_ptr_opencv/src/gnsac_ptr_ocv.cpp
# 	lib/common/src/common.cpp
# 	lib/common/src/comm_loaders.cpp
# 	lib/common/src/comm_math.cpp
# 	lib/common/src/solvers.cpp
# 	lib/gnsac_eigen/src/gnsac_eigen.cpp
# )
# target_link_libraries(det_test_eig ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} stdc++fs)

##################################
#    Create a shared library     #
##################################

add_library(gnsac SHARED
	lib/common/src/common.cpp
	lib/common/src/comm_loaders.cpp
	lib/common/src/comm_math.cpp
	lib/common/src/solvers.cpp
	lib/five_point_opencv/src/five-point.cpp
	lib/five_point_opencv/src/levmarq.cpp
	lib/five_point_opencv/src/preempt.cpp
	lib/five_point_opencv/src/ptsetreg.cpp
	lib/gnsac_ptr_eigen/src/gnsac_ptr_eig.cpp
	lib/gnsac_ptr_opencv/src/gnsac_ptr_ocv.cpp
	lib/gnsac_eigen/src/gnsac_eigen.cpp)

target_include_directories(gnsac PUBLIC
	${EIGEN3_INCLUDE_DIR}
	${YAML_CPP_INCLUDE_DIR}
	${OPENCV_INCLUDE_DIRS}
	lib/common/include
	lib/gnsac_ptr_eigen/include
	lib/gnsac_ptr_opencv/include
	lib/five_point_opencv/include
	lib/gnsac_eigen/include)

target_link_libraries(gnsac ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} stdc++fs)

set(GNSAC_VERSION "0.9")
set(GNSAC_MAJOR_VERSION "0")

set_target_properties(gnsac PROPERTIES
                      VERSION ${GNSAC_VERSION}
                      SOVERSION ${GNSAC_MAJOR_VERSION})

###################
#   CMake Package #
###################

set(CONF_INCLUDE_DIRS 
	${EIGEN3_INCLUDE_DIR}
	${YAML_CPP_INCLUDE_DIR}
	${OPENCV_INCLUDE_DIRS}
	"${PROJECT_SOURCE_DIR}/lib/common/include"
	"${PROJECT_SOURCE_DIR}/lib/gnsac_ptr_eigen/include"
	"${PROJECT_SOURCE_DIR}/lib/gnsac_ptr_opencv/include"
	"${PROJECT_SOURCE_DIR}/lib/five_point_opencv/include"
	"${PROJECT_SOURCE_DIR}/lib/gnsac_eigen/include")
configure_file("cmake/gnsac-config.cmake.in" "${PROJECT_BINARY_DIR}/gnsac-config.cmake" @ONLY)
configure_file("cmake/gnsac-config-version.cmake.in" "${PROJECT_BINARY_DIR}/gnsac-config-version.cmake" @ONLY)

# The EXPORT tells CMake that we want to generate a .cmake
# file that other projects can find using `find_package`
set(INSTALL_LIB_DIR  lib  CACHE PATH "Installation directory for libraries")
install(TARGETS gnsac
    EXPORT  gnsac
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}")
export(EXPORT gnsac)