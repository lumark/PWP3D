cmake_minimum_required( VERSION 2.8 )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra ")
if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

find_package( PWP3D REQUIRED)
include_directories( ${PWP3D_INCLUDE_DIRS})
add_executable( PWP3DAPP PerseusLibTest.cpp )
target_link_libraries(PWP3DAPP ${PWP3D_LIBRARIES})


