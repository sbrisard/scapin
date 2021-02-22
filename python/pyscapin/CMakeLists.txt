find_package(pybind11 REQUIRED)

add_compile_definitions(
        __SCAPIN_AUTHOR__="\"\"${METADATA_AUTHOR}\"\""
        __SCAPIN_VERSION__="\"\"${METADATA_VERSION}\"\""
)

pybind11_add_module(hooke hooke.cpp)
target_link_libraries(hooke PRIVATE scapin)
pybind11_add_module(ms94 ms94.cpp)
target_link_libraries(ms94 PRIVATE scapin)

configure_file(demo.py demo.py COPYONLY)