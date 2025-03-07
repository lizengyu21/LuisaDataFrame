
set(LUISA_DATAFRAME_CORE_SOURCES
    read_csv.cpp)

add_library(luisa-dataframe-io SHARED ${LUISA_DATAFRAME_CORE_SOURCES})

target_link_libraries(luisa-dataframe-io
    PUBLIC
    luisa::compute
    luisa-dataframe-include)
