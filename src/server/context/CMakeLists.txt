add_library(server_context Context.cpp)
target_include_directories(server_context PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/thirdparty/libfiu
    )
