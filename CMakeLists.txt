cmake_minimum_required(VERSION 3.17)
project(lab02 C)

find_package(MPI REQUIRED)
add_compile_options ("-Weverything")

set(CMAKE_C_STANDARD 11)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(lab02 main.c auxil1.c)
target_link_libraries(lab02 ${MPI_C_LIBRARIES})

