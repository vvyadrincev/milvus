add_library(tracing
    TracerUtil.cpp
    TraceContext.cpp
    )

target_include_directories(tracing PUBLIC
    ${INCLUDE_DIRECTORIES}
    )
