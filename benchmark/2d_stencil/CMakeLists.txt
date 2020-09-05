# Copyright (c) 2020 Nikunj Gupta
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.1)

project(2D_Stencil CXX)

find_package(HPX REQUIRED)
link_directories(${HPX_LIBRARY_DIR})
include_directories(${HPX_INCLUDE_DIRS})

add_executable(stencil_parallel_float stencil_parallel_float.cpp)
hpx_setup_target(stencil_parallel_float)
add_executable(stencil_parallel_nfloat stencil_parallel_nfloat.cpp)
hpx_setup_target(stencil_parallel_nfloat)
add_executable(stencil_parallel_ndouble stencil_parallel_ndouble.cpp)
hpx_setup_target(stencil_parallel_ndouble)
add_executable(stencil_parallel_double stencil_parallel_double.cpp)
hpx_setup_target(stencil_parallel_double)

add_executable(stencil_serial stencil_serial.cpp)
hpx_setup_target(stencil_serial)
