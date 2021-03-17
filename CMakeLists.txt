cmake_minimum_required(VERSION 3.17)
project(lab02)

find_package(MPI REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(MPI_hello_world main.cc auxil1.cc auxil1.h utils.h)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
target_link_libraries(MPI_hello_world ${MPI_C_LIBRARIES})

