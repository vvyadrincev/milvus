add_library(cache
    DataObj.h
    Cache.h
    Cache.inl
    CacheMgr.h
    CacheMgr.inl
    CpuCacheMgr.cpp
    GpuCacheMgr.cpp
    )

target_include_directories(cache PUBLIC
    ${INCLUDE_DIRECTORIES}
    )
