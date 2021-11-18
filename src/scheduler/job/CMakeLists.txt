file(GLOB scheduler_job_src
    "*.cpp")

add_library(scheduler_job
    ${scheduler_job_src}
    )

target_include_directories(scheduler_job PUBLIC
    ${INCLUDE_DIRECTORIES}
    )
