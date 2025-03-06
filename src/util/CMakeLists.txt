
set(LUISA_DATAFRAME_CORE_SOURCES
    util.cpp)

add_library(luisa-dataframe-util SHARED ${LUISA_DATAFRAME_CORE_SOURCES})

target_link_libraries(luisa-dataframe-util
    PUBLIC
    luisa::compute
    luisa-dataframe-include)
