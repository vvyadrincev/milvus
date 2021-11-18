# server_config
add_library(server_config
    config.cpp
    )

target_include_directories(server_config PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(server_config PUBLIC
    utils
    config
    )


# server
add_library(server
    server.cpp
    DBWrapper.cpp
    context/Context.cpp
    )

target_include_directories(server PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(server PUBLIC
    server_config
    server_delivery
    0mqServer

    config
    db
    metrics
    scheduler
    tracing
    utils
    wrapper

    Boost::program_options
    nlohmann_json::nlohmann_json
    cppzmq
    ${LIBCBOR_libraries}
    OpenMP::OpenMP_CXX
    )


#server_delivery
add_subdirectory(delivery)
#0mqServer
add_subdirectory(0mq)
