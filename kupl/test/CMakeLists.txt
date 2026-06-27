cmake_minimum_required(VERSION 3.14)

project(kupl_test VERSION 1.0.0)

set(KUPL_INSTALL_BIN_PATH ${CMAKE_INSTALL_PREFIX}/bin)

set(LIBRARY_PATH_ENV "$ENV{LIBRARY_PATH}")
string(REPLACE ":" ";" LIBRARY_PATH_LIST "${LIBRARY_PATH_ENV}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(COMPILER_FILTER "gcc")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(COMPILER_FILTER "clang")
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

if(ENABLE_KUPL_MMA)
    message("-- enable kupl mma")
    add_definitions(-DENABLE_KUPL_MMA)
endif()

set(FILTERED_PATHS "")
foreach(PATH_ITEM IN LISTS LIBRARY_PATH_LIST)
    if(NOT "${PATH_ITEM}" STREQUAL "")
        if("${PATH_ITEM}" MATCHES "${COMPILER_FILTER}")
            list(APPEND FILTERED_PATHS "${PATH_ITEM}")
        endif()
        if("${PATH_ITEM}" MATCHES "gtest")
            list(APPEND FILTERED_PATHS "${PATH_ITEM}")
        endif()
    endif()
endforeach()

find_package(GTest
    HINTS ${FILTERED_PATHS}
)

if("${GTest_FOUND}")
    set(GTEST_LIB "GTest::gtest")
else()
    set(GTEST_LIB "gtest")
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(COV_LIB "gcov")
endif()

add_compile_options(-O0 -g -gdwarf-4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
set(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs -ftest-coverage")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv9-a+sme-f64f64+sve")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv9-a+sme-f64f64+sve")
endif()

file(GLOB_RECURSE TEST_PTHREAD_FILES ./*.cpp)

add_executable(test_main ${TEST_PTHREAD_FILES})
target_link_libraries(test_main
    -Wl,--start-group
    ${GTEST_LIB} kupl dl pthread ${COV_LIB}
    -Wl,--end-group
)
install(TARGETS test_main DESTINATION ${KUPL_INSTALL_BIN_PATH})