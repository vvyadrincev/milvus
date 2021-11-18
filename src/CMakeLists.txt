
add_library(vec_indexer
  indexer.cpp)


set(INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/thirdparty
    ${CMAKE_SOURCE_DIR}/thirdparty/libfiu
    )

target_include_directories(vec_indexer PUBLIC
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(vec_indexer PUBLIC
  faiss
  Boost::boost
  Boost::filesystem
  nlohmann_json::nlohmann_json
  jaegertracing::jaegertracing
  cbor
  )

add_subdirectory(cache)
add_subdirectory(config)
add_subdirectory(db)
add_subdirectory(knowhere)
add_subdirectory(metrics)
add_subdirectory(scheduler)
add_subdirectory(server)
add_subdirectory(storage)
add_subdirectory(tracing)

# Utils
add_subdirectory(utils)



# Wrapper
add_subdirectory(wrapper)

# storage

aux_source_directory(${CMAKE_SOURCE_DIR}/thirdparty/easyloggingpp easyloggingpp_files)

add_executable(vec_indexer_server
    vec_indexer_server.cpp
    ${easyloggingpp_files}
    )

target_link_libraries(vec_indexer_server PUBLIC
  vec_indexer
  server
  )

install(TARGETS vec_indexer_server
        RUNTIME DESTINATION bin)

#CLI
