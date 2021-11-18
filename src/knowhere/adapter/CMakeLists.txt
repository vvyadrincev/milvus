add_library(vector_adapter
    VectorAdapter.cpp
    )

target_include_directories(vector_adapter PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(vector_adapter PUBLIC
    knowhere_common
    )
