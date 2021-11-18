file(GLOB scheduler_task_src
    "*.cpp"
    Task.h
    )

add_library(scheduler_task
    ${scheduler_task_src}
    )

target_include_directories(scheduler_task PUBLIC
    ${INCLUDE_DIRECTORIES}
    )
