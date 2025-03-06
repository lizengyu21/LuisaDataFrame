
set(LUISA_DATAFRAME_CORE_SOURCES
    table.cpp)

add_library(luisa-dataframe-table SHARED ${LUISA_DATAFRAME_CORE_SOURCES})

target_link_libraries(luisa-dataframe-table
    PUBLIC
    luisa::compute
    luisa-dataframe-include)
