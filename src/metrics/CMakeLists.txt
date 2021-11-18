add_library(metrics
    Metrics.cpp
    SystemInfo.cpp
    )

target_include_directories(metrics PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(metrics PUBLIC
    server_config
    nlohmann_json::nlohmann_json
    CUDA::nvml
    )
