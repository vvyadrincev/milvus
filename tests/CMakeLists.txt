
#from https://eb2.co/blog/2015/06/driving-boost.test-with-cmake/
function(add_boost_test _suite_name _src)

    file(READ "${_src}" SOURCE_FILE_CONTENTS)
    #match all test cases
    #exclude commented ones
    string(REGEX MATCHALL
      "\n[^\n/]*(BOOST_FIXTURE_TEST_CASE|BOOST_AUTO_TEST_CASE)\\([\n ]*([A-Za-z_0-9]+)"
      FOUND_TESTS ${SOURCE_FILE_CONTENTS})

    if(NOT FOUND_TESTS)
      #add the whole file
      add_test(NAME "${_suite_name}" COMMAND ${_suite_name})
      return()
    endif()

    foreach(HIT ${FOUND_TESTS})
        string(REGEX REPLACE ".*\\([\n ]*([A-Za-z_0-9]+).*" "\\1" TEST_NAME ${HIT})

        add_test(NAME "${_suite_name}.${TEST_NAME}"
                 COMMAND ${_suite_name} --run_test=${TEST_NAME})
    endforeach()
endfunction()



function(add_exe_and_test
    _name
    _src
    _deps)

  add_executable(${_name} ${_src})
  target_compile_definitions(${_name} PRIVATE BOOST_TEST_DYN_LINK)
  target_link_libraries(${_name} PRIVATE
    Boost::unit_test_framework
    ${_deps})
  target_include_directories(${_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)

  # set_target_properties(${_name} PROPERTIES COMPILE_FLAGS "-fsanitize=address" )
  # set_target_properties(${_name} PROPERTIES LINK_FLAGS "-fsanitize=address" )

  add_boost_test(${_name} ${_src})

endfunction(add_exe_and_test)



#BODY
find_package(Boost 1.68 REQUIRED COMPONENTS unit_test_framework filesystem)


add_exe_and_test(simple_test simple_test.cpp "Boost::filesystem")
