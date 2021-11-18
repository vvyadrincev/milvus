file(GLOB scheduler_action_src
    "*.cpp")

add_library(scheduler_action
    ${scheduler_action_src}
    )

target_include_directories(scheduler_action PUBLIC
    ${INCLUDE_DIRECTORIES}
    )
