cmake_minimum_required(VERSION 3.0)
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

set(pkg_srcs
		src/common/common.cpp
		src/common/comm_loaders.cpp
		src/common/comm_math.cpp
		src/common/solvers.cpp
		src/five_point_opencv/five-point.cpp
		src/five_point_opencv/levmarq.cpp
		src/five_point_opencv/preempt.cpp
		src/five_point_opencv/ptsetreg.cpp		
		src/solvers/gnsac_ptr_ocv.cpp
		src/solvers/gnsac_eigen.cpp
		src/solvers/gnsac_ptr_eig.cpp
		src/parallax_detector/parallax_detector.cpp
	)

##################################
#    Create a shared library     #
##################################



option(SHARED_LIB "Option to build a shared library for use in other projects" ON)
if(SHARED_LIB)

	# Project paths
	set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

	# Define the installation directories and paths
	set(INSTALL_LIB_DIR     lib                       CACHE PATH "Installation directory for libraries")
	set(INSTALL_BIN_DIR     bin                       CACHE PATH "Installation directory for executables")
	set(INSTALL_INCLUDE_DIR include/${PROJECT_NAME}   CACHE PATH "Installation directory for header files")
	set(INSTALL_CMAKE_DIR   lib/cmake/${PROJECT_NAME} CACHE PATH "Installation directory for CMake Config files")


	# Make relative paths absolute (needed later on)
	foreach(p LIB BIN INCLUDE CMAKE)
	    set(var INSTALL_${p}_DIR)
	    if(NOT IS_ABSOLUTE "${${var}}")
	        set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
	    endif()
	endforeach()





	add_library(${PROJECT_NAME} SHARED
		${pkg_srcs}
		)

	target_include_directories(${PROJECT_NAME} 
		PUBLIC
			$<INSTALL_INTERFACE:${INSTALL_INCLUDE_DIR}>
			$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}>
		PRIVATE
			$<INSTALL_INTERFACE:${EIGEN3_INCLUDE_DIR}>
			$<INSTALL_INTERFACE:${YAML_CPP_INCLUDE_DIR}>
			$<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
			$<BUILD_INTERFACE:${YAML_CPP_INCLUDE_DIR}>
		)

	

	target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} stdc++fs)

	set(GNSAC_VERSION "0.9")
	set(GNSAC_MAJOR_VERSION "0")

	set_target_properties(${PROJECT_NAME} PROPERTIES
						VERSION ${GNSAC_VERSION}
						SOVERSION ${GNSAC_MAJOR_VERSION})


	
	##################################
	#  Install library and headers   #
	##################################


	# Install the library to $prefix/${INSTALL_LIB_DIR}  (/usr/local/${INSTALL_LIB_DIR}   in linux)
	install(TARGETS ${PROJECT_NAME} 
		EXPORT ${PROJECT_NAME}
		LIBRARY 
			DESTINATION ${INSTALL_LIB_DIR}
		RUNTIME
			DESTINATION ${INSTALL_LIB_DIR})

	# Install the headers to $prefix/${INSTALL_INCLUDE_DIR}  (/usr/local/${INSTALL_INCLUDE_DIR}   in linux)
	install(DIRECTORY include DESTINATION ${INSTALL_INCLUDE_DIR})

	# Add `make uninstall` target
	configure_file(
	    "${CMAKE_MODULE_PATH}/gnsac_uninstall.cmake.in"
	    "${CMAKE_CURRENT_BINARY_DIR}/gnsac_uninstall.cmake"
	    IMMEDIATE @ONLY)

	add_custom_target(uninstall
	    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/gnsac_uninstall.cmake)


	##################################
	#   CMake Package Installation   #
	##################################

	# Actually place the `find_package` instructions on the user's system
	install(EXPORT ${PROJECT_NAME} DESTINATION "${INSTALL_CMAKE_DIR}")

	# Similiar to the `install(EXPORT ...)` command above, but this creates
	# the gnsac.cmake file that exports all the neccessary CMake info
	# for use of librgnsac out of the build tree.
	export(EXPORT ${PROJECT_NAME})


	# Get the relative path of the file ${INSTALL_INCLUDE_DIR} from the directory ${INSTALL_CMAKE_DIR} and store it in the variable REL_INCLUDE_DIR
	file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")

	set(CONF_INCLUDE_DIRS "\${GNSAC_CMAKE_DIR}/${REL_INCLUDE_DIR}/include/${PROJECT_NAME}")
	configure_file("cmake/gnsac-config.cmake.in" "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gnsac-config.cmake" @ONLY)
	configure_file("cmake/gnsac-config-version.cmake.in" "${PROJECT_BINARY_DIR}/gnsac-config-version.cmake" @ONLY)

	# Install the rransac-config.cmake and rransac-config-version.cmake
	install(FILES
	  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gnsac-config.cmake"
	  "${PROJECT_BINARY_DIR}/gnsac-config-version.cmake"
	  DESTINATION "${INSTALL_CMAKE_DIR}")



##################################
#   Create executable            #
##################################

else()

	include_directories(
		${EIGEN3_INCLUDE_DIR}
		${YAML_CPP_INCLUDE_DIR}
		${OPENCV_INCLUDE_DIRS}
		include/${PROJECT_NAME}
	)

	add_executable(cli
		src/cli.cpp
		${pkg_srcs}
	)
	target_link_libraries(cli ${YAML_CPP_LIBRARIES} ${OpenCV_LIBS} stdc++fs)

endif()




##################################
#   Build Tests                  #
##################################

option(BUILD_TESTS "Build the Params GTests" OFF)
if(BUILD_TESTS)

	if(SHARED_LIB)
  	add_subdirectory(test)
	else()
		message("Must build project as a library to use google tests.")
	endif()
	
endif()