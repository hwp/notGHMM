# - Try to find FLANN C library
# Once done this will define
#  FLANN_FOUND - System has FLANN
#  FLANN_INCLUDE_DIRS - The FLANN include directories
#  FLANN_LIBRARIES - The libraries needed to use FLANN
#  FLANN_DEFINITIONS - Compiler switches required for using FLANN

find_package(PkgConfig)
pkg_check_modules(PC_FLANN QUIET flann)
set(FLANN_DEFINITIONS ${PC_FLANN_CFLAGS_OTHER})

find_path(FLANN_INCLUDE_DIR flann/flann.h
          HINTS ${PC_FLANN_INCLUDEDIR} ${PC_FLANN_INCLUDE_DIRS}
          PATH_SUFFIXES flann)

find_library(FLANN_LIBRARY NAMES flann
             HINTS ${PC_FLANN_LIBDIR} ${PC_FLANN_LIBRARY_DIRS})

set(FLANN_LIBRARIES ${FLANN_LIBRARY})
set(FLANN_INCLUDE_DIRS ${FLANN_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set FLANN_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(FLANN DEFAULT_MSG
                                  FLANN_LIBRARY FLANN_INCLUDE_DIR)

mark_as_advanced(FLANN_INCLUDE_DIR FLANN_LIBRARY)

