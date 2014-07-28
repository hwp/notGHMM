# Author : Weipeng He <heweipeng@gmail.com>
# Copyright (c) 2014, All rights reserved.

cmake_minimum_required(VERSION 2.6)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

project(notGHMM)

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})

set(CMAKE_C_FLAGS "-Wall")

add_subdirectory(src)
add_subdirectory(doc)

