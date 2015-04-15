cmake_minimum_required( VERSION 2.8 )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/")

# Build Configures
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall ")
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# Platform configuration vars
include(SetPlatformVars)

# Overide with cmake -DCMAKE_BUILD_TYPE=Debug {dir}
if( NOT CMAKE_BUILD_TYPE )
   set( CMAKE_BUILD_TYPE Release )
endif()

option(BUILD_SHARED_LIBS "Build Shared Library" ON)
add_subdirectory(PerseusLib)

option(BUILD_PWP3D_APP "Build PWP3D App" OFF)
if(BUILD_PWP3D_APP)
  add_subdirectory(Application)
endif()

