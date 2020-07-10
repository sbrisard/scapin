find_library(MATH_LIBRARY m)

add_library(scapin SHARED
        scapin.cpp
        fft_helper.cpp
        ../include/scapin/scapin.hpp
        ../include/scapin/fft_helper.hpp
        ../include/scapin/hooke.hpp
        ../include/scapin/core.hpp)

target_include_directories(scapin PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}../include>
        $<INSTALL_INTERFACE:include>)

target_include_directories(scapin PUBLIC ../include)
if (MATH_LIBRARY)
    target_link_libraries(scapin ${MATH_LIBRARY})
endif ()

install(TARGETS scapin LIBRARY DESTINATION lib)
install(DIRECTORY ../include/scapin DESTINATION include)
