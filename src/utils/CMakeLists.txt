add_library(utils
    Exception.h
    ThreadPool.h
    BlockingQueue.inl

    LogUtil.cpp
    CommonUtil.cpp
    SignalUtil.cpp
    Status.cpp
    TimeRecorder.cpp
    StringHelpFunctions.cpp
    ValidationUtil.cpp
    )

target_include_directories(utils PRIVATE
    ${INCLUDE_DIRECTORIES}
    )

target_link_libraries(utils PUBLIC
  Boost::program_options
  CUDA::cudart
  )
