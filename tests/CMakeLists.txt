add_executable(test_scapin
        test_scapin_main.cpp
        test_scapin.cpp
        test_fft_helper.cpp
        ../include/scapin/hooke.hpp
        ../include/scapin/core.hpp)

target_include_directories(test_scapin PUBLIC ../include)

if (MATH_LIBRARY)
    target_link_libraries(test_scapin ${MATH_LIBRARY})
endif ()

find_package(Catch2 REQUIRED)
target_link_libraries(test_scapin PRIVATE scapin Catch2::Catch2)

add_test(NAME test_scapin COMMAND test_scapin)
