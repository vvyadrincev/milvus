add_library(db_meta MetaFactory.cpp SqliteMetaImpl.cpp)
target_include_directories(db_meta PUBLIC
    ${INCLUDE_DIRECTORIES}
    )
