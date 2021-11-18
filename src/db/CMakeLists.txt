add_library(db
    DB.h
    Constants.h
    Types.h
    DBFactory.cpp
    DBImpl.cpp
    Utils.cpp
    Options.cpp
    IndexFailedChecker.cpp
    OngoingFileChecker.cpp
    IDGenerator.cpp
    meta/MetaConsts.h
    meta/MetaFactory.cpp
    meta/SqliteMetaImpl.cpp
    )

target_include_directories(db PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(db PUBLIC
    db_insert
    db_engine

    cache
    scheduler
    utils
    vector_index

    faiss
    sqlite3
    )


add_subdirectory(insert)
add_subdirectory(engine)
