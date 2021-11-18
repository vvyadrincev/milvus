add_library(0mqServer
    0mqServer.cpp
    cbor_utils.hpp
    )

target_include_directories(0mqServer PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(0mqServer PUBLIC
    server

    Boost::program_options
    nlohmann_json::nlohmann_json
    cppzmq
    ${LIBCBOR_libraries}
    faiss
    )
