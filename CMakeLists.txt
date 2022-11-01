cmake_minimum_required(VERSION 3.20)

if(DEFINED PROJECT_NAME)
    set(IS_SUBPROJECT true)
endif()

project(treble
    VERSION 1.0.0
    DESCRIPTION "Header-only library for creating self-tuning functions"
    LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
    AddLib
    GIT_REPOSITORY https://github.com/Chemiseblanc/AddLib.git
    GIT_TAG v2.1.2
)
FetchContent_MakeAvailable(AddLib)

include(AddLib)

add_lib(treble
    HEADER_ONLY
    SOURCES
        include/treble/scoped_timer.hpp
        include/treble/self_tuning_function.hpp
        include/treble/treble.hpp
    INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    COMPILE_FEATURES
        cxx_std_17
)

install_project()
package_project(
    CONTACT "Matthew Gibson <matt@mgibson.ca>"
    LICENSE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt"
)

if(NOT IS_SUBPROJECT)
    option(BUILD_EXAMPLES ON)
    if(BUILD_EXAMPLES)
        add_subdirectory(examples)
        add_subdirectory(benchmarks)
    endif()
endif()