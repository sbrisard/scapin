find_library(MATH_LIBRARY m)

find_path(FFTW3_INCLUDE_DIR fftw3.h)
find_library(FFTW3_LIBRARY NAMES libfftw3 libfftw3-3)

find_path(BLITZ_INCLUDE_DIR blitz.h)

find_package(bri17 CONFIG REQUIRED)

add_executable(scapin_demo scapin_demo.cpp)
target_include_directories(scapin_demo PRIVATE ${FFTW3_INCLUDE_DIR})
target_include_directories(scapin_demo PRIVATE ${BLITZ_INCLUDE_DIR})
target_include_directories(scapin_demo PRIVATE ../include)
if (MATH_LIBRARY)
    target_link_libraries(scapin_demo ${MATH_LIBRARY})
endif ()
target_link_libraries(scapin_demo ${FFTW3_LIBRARY})
target_link_libraries(scapin_demo bri17::bri17)
