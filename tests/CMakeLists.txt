find_package(PkgConfig REQUIRED)
pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)

find_package(Catch2 REQUIRED)

add_executable(test_scapin test_scapin.cpp)
target_link_libraries(test_scapin PRIVATE scapin PkgConfig::deps)
add_test(NAME test_scapin COMMAND test_scapin)

add_executable(test_fft_helper test_scapin_main.cpp test_fft_helper.cpp)
target_link_libraries(test_fft_helper PRIVATE scapin Catch2::Catch2)

add_test(NAME test_fft_helper COMMAND test_fft_helper)
