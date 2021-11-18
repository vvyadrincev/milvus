file(GLOB scheduler_resource_src
    "*.cpp")

add_library(scheduler_resource
    ${scheduler_resource_src}
    )

target_include_directories(scheduler_resource PUBLIC
    ${INCLUDE_DIRECTORIES}
    )
