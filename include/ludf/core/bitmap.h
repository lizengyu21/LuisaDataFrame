#pragma once
#include <luisa/luisa-compute.h>
#include <luisa/dsl/syntax.h>
#include <ludf/util/kernel.h>

struct Bitmap {
    luisa::compute::Buffer<uint> _data;
    uint _size = 0u;

    void init(luisa::compute::Device &device, uint size) {
        _data = device.create_buffer<uint>((size + 31u) / 32u);
        _size = size;
    }

    void init(luisa::compute::Device &device, luisa::compute::Stream &stream, uint size, uint init_data) {
        init(device, size);
        stream << ShaderCollector<uint>::get_instance(device)->set_shader(_data, init_data).dispatch(_data.size());
    }

    void init_zero(luisa::compute::Device &device, luisa::compute::Stream &stream, uint size) {
        init(device, size);
        stream << ShaderCollector<uint>::get_instance(device)->reset_shader(_data).dispatch(_data.size());
    }

    void init_one(luisa::compute::Device &device, luisa::compute::Stream &stream, uint size) {
        init(device, size);
        stream << ShaderCollector<uint>::get_instance(device)->set_shader(_data, (0u - 1u)).dispatch(_data.size());
    }
};

LUISA_BINDING_GROUP(Bitmap, _data, _size) {
    void set(const luisa::compute::UInt &index) {
        auto idx = index / 32u;
        auto bit = index % 32u;
        auto mask = 1u << bit;
        auto data = _data->read(idx);
        _data->write(idx, data | mask);
    }

    void clear(const luisa::compute::UInt &index) {
        auto idx = index / 32u;
        auto bit = index % 32u;
        auto mask = ~(1u << bit);
        auto data = _data->read(idx);
        _data->write(idx, data & mask);
    }

    [[nodiscard]] luisa::compute::Bool test(const luisa::compute::UInt &index) const {
        auto idx = index / 32u;
        auto bit = index % 32u;
        auto mask = 1u << bit;
        auto data = _data->read(idx);
        return (data & mask) != 0u;
    }
};