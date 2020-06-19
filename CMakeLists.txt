cmake_minimum_required(VERSION 3.13)
project(
  scapin
  VERSION 0.1
  DESCRIPTION "A framework for FFT-based, full-field numerical simulation of heterogeneous materials."
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
include(CTest)

add_subdirectory(src)
add_subdirectory(tests)
