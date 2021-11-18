# Install script for directory: /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/nix/store/cp1sa3xxvl71cypiinw2c62i5s33chlr-binutils-2.35.1/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/vec_indexer_server" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/vec_indexer_server")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/vec_indexer_server"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/vec_indexer_server")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/vec_indexer_server" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/vec_indexer_server")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/vec_indexer_server"
         OLD_RPATH "/nix/store/f7jpwsi3yiy98pjv6m06m5j2hxds84cm-openssl-1.1.1j/lib:/nix/store/z6mczhdwx4r6krywgylhym58jkcrxcxm-cudatoolkit-11.2.1/lib/stubs:/nix/store/h57yvcannzhpdk1drrc51577l1wqv037-libsodium-1.0.18/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/nix/store/cp1sa3xxvl71cypiinw2c62i5s33chlr-binutils-2.35.1/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/vec_indexer_server")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache/cmake_install.cmake")
  include("/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/config/cmake_install.cmake")
  include("/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/db/cmake_install.cmake")
  include("/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/knowhere/cmake_install.cmake")
  include("/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/metrics/cmake_install.cmake")
  include("/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/scheduler/cmake_install.cmake")
  include("/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/server/cmake_install.cmake")
  include("/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/storage/cmake_install.cmake")
  include("/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/tracing/cmake_install.cmake")
  include("/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/utils/cmake_install.cmake")
  include("/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/wrapper/cmake_install.cmake")

endif()

