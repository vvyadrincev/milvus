file(GLOB db_engine_src
    ExecutionEngine.h
    EngineFactory.cpp
    ExecutionEngineImpl.cpp
    )

add_library(db_engine
    ${db_engine_src}
    )

target_include_directories(db_engine PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(db_engine PUBLIC
    cache
    knowhere_common
    metrics
    scheduler
    server_config
    wrapper
    utils

    faiss
    )
