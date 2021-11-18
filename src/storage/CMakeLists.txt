file(GLOB storage_src
    "*.h"
    "*.cpp"
    "file/*.cpp"
    )

add_library(storage
    ${storage_src}
    )

target_include_directories(storage PUBLIC
    ${INCLUDE_DIRECTORIES}
    )
