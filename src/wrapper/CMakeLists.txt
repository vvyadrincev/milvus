file(GLOB wrapper_src
    "*.cpp"
    "gpu/*.cpp"
    )

add_library(wrapper
    ${wrapper_src}
    )

target_include_directories(wrapper PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(wrapper PUBLIC
    vector_index
    storage
    )
