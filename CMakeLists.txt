cmake_minimum_required(VERSION 3.26...3.29)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)

project(LuisaDataFrame VERSION 0.1.0 LANGUAGES C CXX)

include(CMakeOptions.cmake)
# you may set the LUISA_COMPUTE_ENABLE_XXX options before including LuisaCompute
add_subdirectory(ext/LuisaCompute)
add_subdirectory(ext/csv-parser)

# setup the output directories to make the executables under the same binary directory as LuisaCompute
include(ext/LuisaCompute/scripts/setup_output_dirs.cmake)

file(GLOB_RECURSE LUISA_DATAFRAME_HEADERS CONFIGURE_DEPENDS
    "include/ludf/*.h"
    "include/ludf/*.hpp")
add_library(luisa-dataframe-include INTERFACE ${LUISA_DATAFRAME_HEADERS})
target_include_directories(luisa-dataframe-include INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ext/csv-parser>)
target_compile_features(luisa-dataframe-include INTERFACE c_std_11 cxx_std_20)

add_subdirectory(src)

function(luisa_dataframe_add_executable name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE luisa::compute dataframe)
endfunction ()

add_subdirectory(tests)

