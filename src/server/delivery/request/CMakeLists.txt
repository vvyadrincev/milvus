file(GLOB server_delivery_request_src
    "*.cpp")

add_library(server_delivery_request
    ${server_delivery_request_src}
    )

target_include_directories(server_delivery_request PUBLIC
    ${INCLUDE_DIRECTORIES}
    )
