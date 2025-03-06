
set(LUISA_DATAFRAME_CORE_SOURCES
    type.cpp)

add_library(luisa-dataframe-core SHARED ${LUISA_DATAFRAME_CORE_SOURCES})

target_link_libraries(luisa-dataframe-core
    PUBLIC
    luisa::compute
    luisa-dataframe-include)
