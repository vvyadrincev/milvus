# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists

# Include any dependencies generated for this target.
include src/cache/CMakeFiles/cache.dir/depend.make

# Include the progress variables for this target.
include src/cache/CMakeFiles/cache.dir/progress.make

# Include the compile flags for this target's objects.
include src/cache/CMakeFiles/cache.dir/flags.make

src/cache/CMakeFiles/cache.dir/CpuCacheMgr.cpp.o: src/cache/CMakeFiles/cache.dir/flags.make
src/cache/CMakeFiles/cache.dir/CpuCacheMgr.cpp.o: src/cache/CpuCacheMgr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/cache/CMakeFiles/cache.dir/CpuCacheMgr.cpp.o"
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache && /nix/store/ca37d3qrydh0wpw40kswsx30j8dyzxh2-gcc-wrapper-10.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cache.dir/CpuCacheMgr.cpp.o -c /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache/CpuCacheMgr.cpp

src/cache/CMakeFiles/cache.dir/CpuCacheMgr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cache.dir/CpuCacheMgr.cpp.i"
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache && /nix/store/ca37d3qrydh0wpw40kswsx30j8dyzxh2-gcc-wrapper-10.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache/CpuCacheMgr.cpp > CMakeFiles/cache.dir/CpuCacheMgr.cpp.i

src/cache/CMakeFiles/cache.dir/CpuCacheMgr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cache.dir/CpuCacheMgr.cpp.s"
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache && /nix/store/ca37d3qrydh0wpw40kswsx30j8dyzxh2-gcc-wrapper-10.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache/CpuCacheMgr.cpp -o CMakeFiles/cache.dir/CpuCacheMgr.cpp.s

src/cache/CMakeFiles/cache.dir/GpuCacheMgr.cpp.o: src/cache/CMakeFiles/cache.dir/flags.make
src/cache/CMakeFiles/cache.dir/GpuCacheMgr.cpp.o: src/cache/GpuCacheMgr.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/cache/CMakeFiles/cache.dir/GpuCacheMgr.cpp.o"
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache && /nix/store/ca37d3qrydh0wpw40kswsx30j8dyzxh2-gcc-wrapper-10.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cache.dir/GpuCacheMgr.cpp.o -c /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache/GpuCacheMgr.cpp

src/cache/CMakeFiles/cache.dir/GpuCacheMgr.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cache.dir/GpuCacheMgr.cpp.i"
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache && /nix/store/ca37d3qrydh0wpw40kswsx30j8dyzxh2-gcc-wrapper-10.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache/GpuCacheMgr.cpp > CMakeFiles/cache.dir/GpuCacheMgr.cpp.i

src/cache/CMakeFiles/cache.dir/GpuCacheMgr.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cache.dir/GpuCacheMgr.cpp.s"
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache && /nix/store/ca37d3qrydh0wpw40kswsx30j8dyzxh2-gcc-wrapper-10.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache/GpuCacheMgr.cpp -o CMakeFiles/cache.dir/GpuCacheMgr.cpp.s

# Object files for target cache
cache_OBJECTS = \
"CMakeFiles/cache.dir/CpuCacheMgr.cpp.o" \
"CMakeFiles/cache.dir/GpuCacheMgr.cpp.o"

# External object files for target cache
cache_EXTERNAL_OBJECTS =

src/cache/libcache.a: src/cache/CMakeFiles/cache.dir/CpuCacheMgr.cpp.o
src/cache/libcache.a: src/cache/CMakeFiles/cache.dir/GpuCacheMgr.cpp.o
src/cache/libcache.a: src/cache/CMakeFiles/cache.dir/build.make
src/cache/libcache.a: src/cache/CMakeFiles/cache.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libcache.a"
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache && $(CMAKE_COMMAND) -P CMakeFiles/cache.dir/cmake_clean_target.cmake
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cache.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/cache/CMakeFiles/cache.dir/build: src/cache/libcache.a

.PHONY : src/cache/CMakeFiles/cache.dir/build

src/cache/CMakeFiles/cache.dir/clean:
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache && $(CMAKE_COMMAND) -P CMakeFiles/cache.dir/cmake_clean.cmake
.PHONY : src/cache/CMakeFiles/cache.dir/clean

src/cache/CMakeFiles/cache.dir/depend:
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/cache/CMakeFiles/cache.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/cache/CMakeFiles/cache.dir/depend

