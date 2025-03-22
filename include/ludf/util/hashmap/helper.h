#pragma once
#include <ludf/core/hashmap.h>
#include <ludf/util/kernel.h>
#include <luisa/luisa-compute.h>

template <class T>
inline void init(luisa::compute::Device &device, luisa::compute::Stream &stream, Hashmap<T> &hashmap, uint capacity) {
    using namespace luisa;
    using namespace luisa::compute;

    hashmap._capacity = capacity;
    hashmap._counter = device.create_buffer<uint>(capacity);
    hashmap._offset = device.create_buffer<uint>(capacity);
    hashmap._key = device.create_buffer<BaseType>(capacity * sizeof(T) / sizeof(BaseType));

    stream << ShaderCollector<uint>::get_instance(device)->set_shader(hashmap._counter, 0u).dispatch(capacity);
    stream << ShaderCollector<uint>::get_instance(device)->set_shader(hashmap._offset, 0u).dispatch(capacity);
    stream << ShaderCollector<BaseType>::get_instance(device)->set_shader(hashmap._key, BASE_NULL).dispatch(capacity);
}