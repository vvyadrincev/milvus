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
include src/server/CMakeFiles/server_config.dir/depend.make

# Include the progress variables for this target.
include src/server/CMakeFiles/server_config.dir/progress.make

# Include the compile flags for this target's objects.
include src/server/CMakeFiles/server_config.dir/flags.make

src/server/CMakeFiles/server_config.dir/config.cpp.o: src/server/CMakeFiles/server_config.dir/flags.make
src/server/CMakeFiles/server_config.dir/config.cpp.o: src/server/config.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/server/CMakeFiles/server_config.dir/config.cpp.o"
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/server && /nix/store/ca37d3qrydh0wpw40kswsx30j8dyzxh2-gcc-wrapper-10.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/server_config.dir/config.cpp.o -c /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/server/config.cpp

src/server/CMakeFiles/server_config.dir/config.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/server_config.dir/config.cpp.i"
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/server && /nix/store/ca37d3qrydh0wpw40kswsx30j8dyzxh2-gcc-wrapper-10.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/server/config.cpp > CMakeFiles/server_config.dir/config.cpp.i

src/server/CMakeFiles/server_config.dir/config.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/server_config.dir/config.cpp.s"
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/server && /nix/store/ca37d3qrydh0wpw40kswsx30j8dyzxh2-gcc-wrapper-10.2.0/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/server/config.cpp -o CMakeFiles/server_config.dir/config.cpp.s

# Object files for target server_config
server_config_OBJECTS = \
"CMakeFiles/server_config.dir/config.cpp.o"

# External object files for target server_config
server_config_EXTERNAL_OBJECTS =

src/server/libserver_config.a: src/server/CMakeFiles/server_config.dir/config.cpp.o
src/server/libserver_config.a: src/server/CMakeFiles/server_config.dir/build.make
src/server/libserver_config.a: src/server/CMakeFiles/server_config.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libserver_config.a"
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/server && $(CMAKE_COMMAND) -P CMakeFiles/server_config.dir/cmake_clean_target.cmake
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/server && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/server_config.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/server/CMakeFiles/server_config.dir/build: src/server/libserver_config.a

.PHONY : src/server/CMakeFiles/server_config.dir/build

src/server/CMakeFiles/server_config.dir/clean:
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/server && $(CMAKE_COMMAND) -P CMakeFiles/server_config.dir/cmake_clean.cmake
.PHONY : src/server/CMakeFiles/server_config.dir/clean

src/server/CMakeFiles/server_config.dir/depend:
	cd /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/server /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/server /home/exuser/develop/vec_indexer_milvus_but_new_cmakelists/src/server/CMakeFiles/server_config.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/server/CMakeFiles/server_config.dir/depend

