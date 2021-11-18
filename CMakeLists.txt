
cmake_minimum_required(VERSION 3.12)

project(VecIndexer VERSION 0.1
  LANGUAGES CXX)

#devel
set( CMAKE_EXPORT_COMPILE_COMMANDS 1 )

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_CLANG_TIDY "Perform Clang-Tidy diagnostics" OFF)
option(CLANG_TIDY_FIX "Perform fixes for Clang-Tidy" OFF)
find_program(
  CLANG_TIDY_EXE
  NAMES "clang-tidy"
  DOC "Path to clang-tidy executable"
  )

if(ENABLE_CLANG_TIDY AND CLANG_TIDY_EXE)
  message("CLANG_TIDY_EXE is found: ${CLANG_TIDY_EXE}")
  if(CLANG_TIDY_FIX)
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}" "-fix")
  else()
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}"
      "-checks='*,-fuchsia-*,-google-*,\
-hicpp-braces-around-statements,\
-modernize-use-trailing-return-type,-modernize-use-nodiscard,\
-misc-non-private-member-variables-in-classes,-misc-unused-parameters,\
-readability-magic-numbers,-readability-isolate-declaration,\
-cppcoreguidelines-avoid-magic-numbers")

  endif()
endif()


set(MILVUS_GPU_VERSION true)
if (MILVUS_GPU_VERSION)
    message(STATUS "Building GPU version")
    add_compile_definitions("MILVUS_GPU_VERSION")
endif ()



find_package(Boost 1.68 REQUIRED COMPONENTS program_options filesystem)
find_package(faiss REQUIRED)
find_package(cppzmq REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(jaegertracing REQUIRED)
find_package(OpenMP REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBCBOR REQUIRED libcbor)
find_package(CUDAToolkit REQUIRED COMPONENTS cudart nvml)

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
