# Author : Weipeng He <heweipeng@gmail.com>
# Copyright (c) 2014, All rights reserved.

cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(notGHMM)

set (PACKAGE "notghmm")
set (PACKAGE_NAME "NotGHMM")
set (PACKAGE_VERSION "1.0.0")
set (PACKAGE_STRING "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set (PACKAGE_URL "http://www.github.com/hwp/notghmm")

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})

set(CMAKE_C_FLAGS "-Wall")

find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

add_subdirectory(notghmm)
add_subdirectory(doc)

if (UNIX)
  configure_file(${CMAKE_SOURCE_DIR}/notghmm.pc.cmake.in ${CMAKE_BINARY_DIR}/notghmm.pc @ONLY)
  install(FILES ${CMAKE_BINARY_DIR}/notghmm.pc DESTINATION lib/pkgconfig)
endif()

