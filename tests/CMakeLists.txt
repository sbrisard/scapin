find_package(Catch2 REQUIRED)

add_executable(test_scapin test_scapin_main.cpp test_scapin.cpp test_fft_helper.cpp)
target_link_libraries(test_scapin PRIVATE scapin Catch2::Catch2)

add_test(NAME test_scapin COMMAND test_scapin)
