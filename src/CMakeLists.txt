add_library(dataframe INTERFACE)

add_subdirectory(core)
add_subdirectory(util)
add_subdirectory(table)

target_link_libraries(dataframe INTERFACE
    luisa-dataframe-core
    luisa-dataframe-util
    luisa-dataframe-table)