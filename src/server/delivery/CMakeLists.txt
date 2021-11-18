add_subdirectory(request)

add_library(server_delivery
    RequestHandler.cpp
    RequestScheduler.cpp
    )

target_include_directories(server_delivery PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(server_delivery PUBLIC
    server_delivery_request
    )
