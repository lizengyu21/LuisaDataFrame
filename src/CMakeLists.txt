add_library(dataframe INTERFACE)

add_subdirectory(core)
add_subdirectory(util)
add_subdirectory(table)
add_subdirectory(io)

target_link_libraries(dataframe INTERFACE
    luisa-dataframe-core
    luisa-dataframe-util
    luisa-dataframe-table
    luisa-dataframe-io)