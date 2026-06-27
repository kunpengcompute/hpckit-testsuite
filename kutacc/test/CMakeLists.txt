cmake_minimum_required(VERSION 3.14)

project(kutacc_test VERSION 1.0.0)

set(KUTACC_INSTALL_BIN_PATH ${CMAKE_INSTALL_PREFIX}/bin)

set(LIBRARY_PATH_ENV "$ENV{LIBRARY_PATH}")
string(REPLACE ":" ";" LIBRARY_PATH_LIST "${LIBRARY_PATH_ENV}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(COMPILER_FILTER "gcc")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(COMPILER_FILTER "clang")
else()
    message(FATAL_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

set(FILTERED_PATHS "")
foreach(PATH_ITEM IN LISTS LIBRARY_PATH_LIST)
    if(NOT "${PATH_ITEM}" STREQUAL "")
        if("${PATH_ITEM}" MATCHES "${COMPILER_FILTER}")
            list(APPEND FILTERED_PATHS "${PATH_ITEM}")
        endif()
    endif()
endforeach()

find_package(GTest
    HINTS ${FILTERED_PATHS}
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(COV_LIB "gcov")
endif()

add_compile_options(-O0 -g -gdwarf-4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage -march=armv8.6-a+sme-f64f64+fp16+bf16 -msve-vector-bits=512")
set(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs -ftest-coverage")

file(GLOB_RECURSE TEST_KUTACC_FILES *.cpp)

add_executable(test_kutacc ${TEST_KUTACC_FILES})
target_link_libraries(test_kutacc
    -Wl,--start-group
    kutacc_core ${COV_LIB}
    -Wl,--end-group
)
install(TARGETS test_kutacc DESTINATION ${KUTACC_INSTALL_BIN_PATH})