cmake_minimum_required( VERSION 2.8 )
project( PWP3D )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/")

# Build Configures
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wno-unused-local-typedefs -Wno-unused-parameter -Wno-unknown-pragmas -Wno-deprecated-declarations")
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

option(BUILD_APPLICATIONS "Build Applications" ON)
option(BUILD_SHARED_LIBS "Build Shared Library" ON)

# Overide with cmake -DCMAKE_BUILD_TYPE=Debug {dir}
if( NOT CMAKE_BUILD_TYPE )
   set( CMAKE_BUILD_TYPE Release )
endif()

# Platform configuration vars
include(SetPlatformVars)

add_subdirectory(PerseusLib)

#if(BUILD_APPLICATIONS)
#  add_subdirectory(applications)
#endif()

