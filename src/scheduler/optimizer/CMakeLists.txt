file(GLOB scheduler_optimizer_src
    "*.cpp")

add_library(scheduler_optimizer
    ${scheduler_optimizer_src}
    )

target_include_directories(scheduler_optimizer PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(scheduler_optimizer PUBLIC
    server_config
    )
