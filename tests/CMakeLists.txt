find_package(PkgConfig REQUIRED)
pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)

add_executable(test_scapin test_scapin.c)
target_link_libraries(test_scapin PRIVATE scapin PkgConfig::deps)
add_test(NAME test_scapin COMMAND test_scapin)

add_executable(test_fft_helper test_fft_helper.cpp)
target_link_libraries(test_fft_helper PRIVATE scapin PkgConfig::deps)

add_test(NAME test_fft_helper COMMAND test_fft_helper)
