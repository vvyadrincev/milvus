add_library(knowhere_common
    Exception.cpp
    Dataset.h
    BinarySet.h
    )

target_include_directories(knowhere_common PUBLIC
    ${INCLUDE_DIRECTORIES}
    )
