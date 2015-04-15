cmake_minimum_required( VERSION 2.8 )
project( PWP3D )

set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR})
string( TOLOWER ${PROJECT_NAME} LIBRARY_NAME )

################################################################################

list(APPEND SRC_H
CUDA/CUDAConvolution.h
CUDA/CUDAData.h
CUDA/CUDADT.h
CUDA/CUDAEF.h
CUDA/CUDAEngine.h
CUDA/CUDARenderer.h
CUDA/CUDAScharr.h
CUDA/CUDAUtils.h 

Objects/HistogramVarBin.h
Objects/ImageRender.h
Objects/IterationConfiguration.h
Objects/Object3D.h
Objects/Object3DParams.h
Objects/Pose3D.h
Objects/StepSize3D.h
Objects/View3D.h
Objects/View3DParams.h

Optimiser/EFs/EFStandard.h
Optimiser/EFs/IEnergyFunction.h
Optimiser/Engine/OptimisationEngine.h

Others/PerseusLibDefines.h

PerseusLib.h

Primitives/ImagePerseus.h
Primitives/PixelUCHAR4.h
Primitives/Vector2D.h
Primitives/Vector3D.h
Primitives/Vector4D.h

Renderer/Engine/DrawingEngine.h
Renderer/Engine/DrawingPrimitives.h

Renderer/Model/Model.h
Renderer/Model/ModelFace.h
Renderer/Model/ModelGroup.h
Renderer/Model/ModelH.h
Renderer/Model/ModelVertex.h
Renderer/Objects/Renderer3DObject.h
Renderer/Objects/Renderer3DView.h
Renderer/Primitives/Camera3D.h
Renderer/Primitives/Quaternion.h
Renderer/Transforms/CameraCoordinateTransform.h
Renderer/Transforms/CoordinateTransform.h
Renderer/Transforms/ObjectCoordinateTransform.h

Utils/Debug.h
Utils/FileUtils.h
Utils/HistogramEngine.h
Utils/ImageUtils.h
Utils/MathUtils.h
Utils/VisualisationEngine.h
)


list(APPEND SRC_CU
CUDA/CUDAConvolution.cu
CUDA/CUDADT.cu
CUDA/CUDAEF.cu
CUDA/CUDAEngine.cu
CUDA/CUDARenderer.cu
CUDA/CUDAScharr.cu
CUDA/CUDAUtils.cu

Optimiser/EFs/EFStandard.cpp
Optimiser/Engine/OptimisationEngine.cpp

Renderer/Engine/DrawingEngine.cpp
Renderer/Engine/DrawingPrimitives.cpp
Renderer/Model/Model.cpp
Renderer/Model/ModelGroup.cpp
Renderer/Model/ModelH.cpp
Renderer/Primitives/Quaternion.cpp
Renderer/Transforms/CameraCoordinateTransform.cpp
Renderer/Transforms/CoordinateTransform.cpp

Utils/FileUtils.cpp
Utils/HistogramEngine.cpp
Utils/ImageUtils.cpp
Utils/MathUtils.cpp
Utils/VisualisationEngine.cpp
)


################################################################################
# Find required dependencies

# Find CUDA toolkit. Thrust and NPP are bundled with CUDA toolkit.
find_package(CUDA REQUIRED)
list( APPEND USER_INC ${CUDA_INCLUDE_DIRS} )
set(HAVE_THRUST 1)
set(HAVE_NPP 1)

# Dont propogate host flags into CUDA environment.
set(CUDA_PROPAGATE_HOST_FLAGS off)

# Generate code for 2.0 and 3.0 CUDA compute architectures (TODO: Verify this works)
set(CUDA_NVCC_FLAGS "-use_fast_math -gencode arch=compute_30,code=sm_30 -gencode arch=compute_20,code=sm_20" )

# Make sure that NPP is found
if(NOT CUDA_npp_LIBRARY)
    # TODO: Fix FindCUDA or make this more robust
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CUDA_npp_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/libnppc.dylib;${CUDA_TOOLKIT_ROOT_DIR}/lib/libnppi.dylib;${CUDA_TOOLKIT_ROOT_DIR}/lib/libnpps.dylib")
    elseif(WIN32)
        set(CUDA_npp_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/nppc.lib;${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/nppi.lib;${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32/npps.lib")
    elseif(WIN64)
        set(CUDA_npp_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nppc.lib;${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nppi.lib;${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/npps.lib")
    else()
        set(CUDA_npp_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnpps.so;${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnppi.so;${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnpps.so")
    endif()
endif()
list(APPEND LINK_LIBS ${CUDA_npp_LIBRARY} )

find_package( Eigen3 QUIET )
if(EIGEN3_FOUND)
    set(HAVE_EIGEN 1)
    list( APPEND USER_INC ${EIGEN3_INCLUDE_DIR} )
endif()


find_package( FREEIMAGE REQUIRED )
if(FREEIMAGE_FOUND)
    set(HAVE_FREEIMAGE 1)
    list(APPEND USER_INC ${FREEIMAGE_INCLUDE_DIRS} )
    list(APPEND LINK_LIBS ${FREEIMAGE_LIBRARIES} )
endif()

find_package( ASSIMP REQUIRED )
if(ASSIMP_FOUND)
    set(HAVE_ASSIMP 1)
    list(APPEND USER_INC ${ASSIMP_INCLUDE_DIR})
    list(APPEND LINK_LIBS ${ASSIMP_LIBRARIES} )
endif()

find_package( OpenCV REQUIRED )
if(FREEIMAGE_FOUND)
    set(HAVE_OpenCV 1)
    list(APPEND USER_INC ${OpenCV_INCLUDE_DIRS} )
    list(APPEND LINK_LIBS ${OpenCV_LIBRARIES} )
endif()

# So that we can find config.h
list( APPEND LIB_INC_DIR "${CMAKE_CURRENT_BINARY_DIR}/..;${CMAKE_CURRENT_SOURCE_DIR}/.." )

#######################################################
## Create configure file for inclusion in library

CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/config.h"
)

################################################################################
# Include collected includes / libraries
include_directories( ${USER_INC} )
include_directories( ${LIB_INC_DIR} )
include_directories( ${INTERNAL_INC} )
cuda_add_library( ${LIBRARY_NAME} ${SRC_H} ${SRC_CU} )
target_link_libraries(${LIBRARY_NAME} ${LINK_LIBS})

## Generate symbol export helper header on MSVC
if(MSVC)
    string(TOUPPER ${LIBRARY_NAME} LIBRARY_NAME_CAPS)
    include(GenerateExportHeader)
    generate_export_header( ${LIBRARY_NAME}
        BASE_NAME ${LIBRARY_NAME_CAPS}
        EXPORT_MACRO_NAME ${LIBRARY_NAME_CAPS}_EXPORT
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/${LIBRARY_NAME}_export.h"
        STATIC_DEFINE ${LIBRARY_NAME_CAPS}_BUILT_AS_STATIC
    )
endif()

#######################################################

# This relative path allows installed files to be relocatable.
set( CMAKECONFIG_INSTALL_DIR "lib/cmake/${PROJECT_NAME}" )
file( RELATIVE_PATH REL_INCLUDE_DIR
    "${CMAKE_INSTALL_PREFIX}/${CMAKECONFIG_INSTALL_DIR}"
    "${CMAKE_INSTALL_PREFIX}/include" )

# Export library for easy inclusion from other cmake projects.
export( TARGETS ${LIBRARY_NAME}
        FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake" )

# Version information
configure_file("${PROJECT_NAME}ConfigVersion.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake" @ONLY)

# Build tree config
set( EXPORT_LIB_INC_DIR "${LIB_INC_DIR}" )
CONFIGURE_FILE( "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" @ONLY IMMEDIATE )

# Install tree config
set( EXPORT_LIB_INC_DIR "\${${PROJECT_NAME}_CMAKE_DIR}/${REL_INCLUDE_DIR}" )
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake" @ONLY )

# Add package to CMake package registery for use from the build tree
export( PACKAGE ${PROJECT_NAME} )

#######################################################
## Install headers / targets

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/config.h"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(FILES ${SRC_H}
  DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}
)
install(TARGETS ${LIBRARY_NAME}
  EXPORT "${PROJECT_NAME}Targets"
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

#######################################################
## Install CMake config

INSTALL(
    FILES "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION ${CMAKECONFIG_INSTALL_DIR} )

install( EXPORT "${PROJECT_NAME}Targets" DESTINATION ${CMAKECONFIG_INSTALL_DIR} )
