file(GLOB vector_index_src
    "Faiss*.cpp"
    "Index*IVF.cpp"
    "Index*IVFPQ.cpp"
    "Index*IDMAP.cpp"
    VectorIndex.h
    ../preprocessor/Preprocessor.h
    )

add_library(vector_index
    ${vector_index_src})

target_include_directories(vector_index PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(vector_index PUBLIC
    vector_index_helpers
    vector_adapter
    knowhere_common

    faiss
    )

add_subdirectory(helpers)
