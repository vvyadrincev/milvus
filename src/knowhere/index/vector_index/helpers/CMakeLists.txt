add_library(vector_index_helpers
    FaissGpuResourceMgr.cpp
    IndexParameter.cpp
    Cloner.cpp
    FaissIO.cpp
    )

target_include_directories(vector_index_helpers PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(vector_index_helpers PUBLIC
    knowhere_common
    utils

    faiss
    )
