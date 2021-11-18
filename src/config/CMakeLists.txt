add_library(config
    ConfigNode.cpp
    YamlConfigMgr.cpp
    )

target_include_directories(config PUBLIC
    ${INCLUDE_DIRECTORIES}
    )
