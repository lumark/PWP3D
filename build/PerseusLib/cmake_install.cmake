# Install script for directory: /Users/luma/Code/Luma/PWP3D/PerseusLib

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/include/PWP3D/config.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/include/PWP3D" TYPE FILE FILES "/Users/luma/Code/Luma/PWP3D/build/PerseusLib/config.h")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/include/PWP3D/CUDAConvolution.h;/usr/local/include/PWP3D/CUDAData.h;/usr/local/include/PWP3D/CUDADT.h;/usr/local/include/PWP3D/CUDAEF.h;/usr/local/include/PWP3D/CUDAEngine.h;/usr/local/include/PWP3D/CUDARenderer.h;/usr/local/include/PWP3D/CUDAScharr.h;/usr/local/include/PWP3D/CUDAUtils.h;/usr/local/include/PWP3D/HistogramVarBin.h;/usr/local/include/PWP3D/ImageRender.h;/usr/local/include/PWP3D/IterationConfiguration.h;/usr/local/include/PWP3D/Object3D.h;/usr/local/include/PWP3D/Object3DParams.h;/usr/local/include/PWP3D/Pose3D.h;/usr/local/include/PWP3D/StepSize3D.h;/usr/local/include/PWP3D/View3D.h;/usr/local/include/PWP3D/View3DParams.h;/usr/local/include/PWP3D/EFStandard.h;/usr/local/include/PWP3D/IEnergyFunction.h;/usr/local/include/PWP3D/OptimisationEngine.h;/usr/local/include/PWP3D/PerseusLibDefines.h;/usr/local/include/PWP3D/PerseusLib.h;/usr/local/include/PWP3D/ImagePerseus.h;/usr/local/include/PWP3D/PixelUCHAR4.h;/usr/local/include/PWP3D/Vector2D.h;/usr/local/include/PWP3D/Vector3D.h;/usr/local/include/PWP3D/Vector4D.h;/usr/local/include/PWP3D/DrawingEngine.h;/usr/local/include/PWP3D/DrawingPrimitives.h;/usr/local/include/PWP3D/Model.h;/usr/local/include/PWP3D/ModelFace.h;/usr/local/include/PWP3D/ModelGroup.h;/usr/local/include/PWP3D/ModelH.h;/usr/local/include/PWP3D/ModelVertex.h;/usr/local/include/PWP3D/Renderer3DObject.h;/usr/local/include/PWP3D/Renderer3DView.h;/usr/local/include/PWP3D/Camera3D.h;/usr/local/include/PWP3D/Quaternion.h;/usr/local/include/PWP3D/CameraCoordinateTransform.h;/usr/local/include/PWP3D/CoordinateTransform.h;/usr/local/include/PWP3D/ObjectCoordinateTransform.h;/usr/local/include/PWP3D/Debug.h;/usr/local/include/PWP3D/FileUtils.h;/usr/local/include/PWP3D/HistogramEngine.h;/usr/local/include/PWP3D/ImageUtils.h;/usr/local/include/PWP3D/MathUtils.h;/usr/local/include/PWP3D/VisualisationEngine.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/include/PWP3D" TYPE FILE FILES
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/CUDA/CUDAConvolution.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/CUDA/CUDAData.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/CUDA/CUDADT.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/CUDA/CUDAEF.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/CUDA/CUDAEngine.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/CUDA/CUDARenderer.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/CUDA/CUDAScharr.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/CUDA/CUDAUtils.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Objects/HistogramVarBin.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Objects/ImageRender.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Objects/IterationConfiguration.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Objects/Object3D.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Objects/Object3DParams.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Objects/Pose3D.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Objects/StepSize3D.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Objects/View3D.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Objects/View3DParams.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Optimiser/EFs/EFStandard.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Optimiser/EFs/IEnergyFunction.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Optimiser/Engine/OptimisationEngine.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Others/PerseusLibDefines.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/PerseusLib.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Primitives/ImagePerseus.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Primitives/PixelUCHAR4.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Primitives/Vector2D.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Primitives/Vector3D.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Primitives/Vector4D.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Engine/DrawingEngine.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Engine/DrawingPrimitives.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Model/Model.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Model/ModelFace.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Model/ModelGroup.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Model/ModelH.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Model/ModelVertex.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Objects/Renderer3DObject.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Objects/Renderer3DView.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Primitives/Camera3D.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Primitives/Quaternion.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Transforms/CameraCoordinateTransform.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Transforms/CoordinateTransform.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Renderer/Transforms/ObjectCoordinateTransform.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Utils/Debug.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Utils/FileUtils.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Utils/HistogramEngine.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Utils/ImageUtils.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Utils/MathUtils.h"
    "/Users/luma/Code/Luma/PWP3D/PerseusLib/Utils/VisualisationEngine.h"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/lib/libpwp3d.dylib")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/lib" TYPE SHARED_LIBRARY FILES "/Users/luma/Code/Luma/PWP3D/build/PerseusLib/libpwp3d.dylib")
  if(EXISTS "$ENV{DESTDIR}/usr/local/lib/libpwp3d.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/lib/libpwp3d.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/usr/local/cuda/lib"
      "$ENV{DESTDIR}/usr/local/lib/libpwp3d.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/usr/local/lib/libpwp3d.dylib")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/PWP3D" TYPE FILE FILES
    "/Users/luma/Code/Luma/PWP3D/build/PerseusLib/CMakeFiles/PWP3DConfig.cmake"
    "/Users/luma/Code/Luma/PWP3D/build/PerseusLib/PWP3DConfigVersion.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/PWP3D/PWP3DTargets.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/PWP3D/PWP3DTargets.cmake"
         "/Users/luma/Code/Luma/PWP3D/build/PerseusLib/CMakeFiles/Export/lib/cmake/PWP3D/PWP3DTargets.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/PWP3D/PWP3DTargets-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/PWP3D/PWP3DTargets.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/PWP3D" TYPE FILE FILES "/Users/luma/Code/Luma/PWP3D/build/PerseusLib/CMakeFiles/Export/lib/cmake/PWP3D/PWP3DTargets.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/PWP3D" TYPE FILE FILES "/Users/luma/Code/Luma/PWP3D/build/PerseusLib/CMakeFiles/Export/lib/cmake/PWP3D/PWP3DTargets-release.cmake")
  endif()
endif()

