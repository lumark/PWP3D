#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "pwp3d" for configuration "Release"
set_property(TARGET pwp3d APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pwp3d PROPERTIES
  IMPORTED_LOCATION_RELEASE "/usr/local/lib/libpwp3d.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libpwp3d.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS pwp3d )
list(APPEND _IMPORT_CHECK_FILES_FOR_pwp3d "/usr/local/lib/libpwp3d.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
