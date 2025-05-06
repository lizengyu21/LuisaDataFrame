#pragma once
#include <luisa/luisa-compute.h>
#include <luisa/dsl/syntax.h>
#include <ludf/util/kernel.h>

struct Bitmap {
    luisa::compute::Buffer<uint> _data;
    uint _size = 0u;

    void init(luisa::compute::Device &device, uint size) {
        _data = device.create_buffer<uint>(size / 32u + 1);
        _size = size;
    }

    void init_zero(luisa::compute::Device &device, luisa::compute::Stream &stream, uint size, auto &&shader) {
        init(device, size);
        stream << shader(_data, 0u).dispatch(_data.size());
    }

    Bitmap copy(luisa::compute::Device &device, luisa::compute::Stream &stream) {
        Bitmap tmp;
        if (_data.size() == 0) return std::move(tmp);
        tmp._data = device.create_buffer<uint>(_data.size());
        tmp._size = _size;
        stream << tmp._data.copy_from(_data);
        return std::move(tmp);
    }

    // void init_one(luisa::compute::Device &device, luisa::compute::Stream &stream, uint size) {
    //     init(device, size);
    //     stream << ShaderCollector<uint>::get_instance(device)->set_shader(_data, (0u - 1u)).dispatch(_data.size());
    // }
};

LUISA_BINDING_GROUP(Bitmap, _data, _size) {
    void set(const luisa::compute::UInt &index) {
        auto idx = index / 32u;
        auto bit = index % 32u;
        auto mask = 1u << bit;
        _data->atomic(idx).fetch_or(mask);
    }

    void clear(const luisa::compute::UInt &index) {
        auto idx = index / 32u;
        auto bit = index % 32u;
        auto mask = ~(1u << bit);
        _data->atomic(idx).fetch_and(mask);
    }

    [[nodiscard]] luisa::compute::Bool test(const luisa::compute::UInt &index) const {
        
        auto idx = index / 32u;
        auto bit = index % 32u;
        auto mask = 1u << bit;
        auto data = _data->read(idx);
        return (data & mask) != 0u;
    }
};