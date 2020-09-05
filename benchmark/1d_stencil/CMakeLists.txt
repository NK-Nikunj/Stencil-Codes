# Copyright (c) 2020 Nikunj Gupta
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.10)

project(stencil CXX)

find_package(HPX REQUIRED)
link_directories(${HPX_LIBRARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(1d_stencil 1d_stencil.cpp)
hpx_setup_target(1d_stencil)
