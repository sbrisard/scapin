cmake_minimum_required(VERSION 3.13)
project(scapin C)

set(CMAKE_C_STANDARD 11)
include(CTest)

add_subdirectory(src)
add_subdirectory(tests)
