file(GLOB db_insert_src
    "*.cpp"
    "../Constants.h"
    "../meta/Meta.h"
    "MemManager.h"
    )

add_library(db_insert
    ${db_insert_src}
    )

target_include_directories(db_insert PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(db_insert PUBLIC
    db_engine

    metrics
    utils
    )
