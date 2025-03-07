#pragma once
#include <ludf/core/type_dispatcher.h>
#include <luisa/luisa-compute.h>
#include <ludf/util/kernel.h>

class Column {
    static const size_t stride = sizeof(BaseType);
public:
    DataType _dtype;
    BufferBase _data;

    Column(DataType dtype = DataType{TypeId::EMPTY}) : _dtype(dtype) {}
    Column(BufferBase &&data, DataType dtype) : _dtype(dtype), _data(std::move(data)) {}
    Column(Column &&) = default;
    Column(const Column &) = delete;
    Column &operator=(Column &&) = default;

    template <class T>
    luisa::compute::BufferView<T> view() {
        return _data.view().as<T>();
    }

    size_t size_bytes() const {
        return _data.size_bytes();
    }

    size_t size() const {
        if (_dtype.id() == TypeId::EMPTY) return 0;
        return _data.size_bytes() / id_to_size(_dtype.id());
    }

    DataType dtype() const {
        return _dtype;
    }

    void set_dtype(DataType dtype) {
        _dtype = dtype;
    }

    Column clone(luisa::compute::Device &device, luisa::compute::Stream &stream) const {
        BufferBase t = device.create_buffer<BaseType>(_data.size());
        stream << t.copy_from(_data);
        return std::move(Column{std::move(t), _dtype});
    }

    void resize(luisa::compute::Device &device, luisa::compute::Stream &stream, size_t size_byte) {
        if (size_byte == 0) {
            _data = BufferBase();
            return;
        }
        auto t = device.create_buffer<BaseType>(size_byte / stride);
        size_t dispatch_size = t.size() < _data.size() ? t.size() : _data.size();
        stream << ShaderCollector<BaseType>::get_instance(device)->copy_shader(t, _data).dispatch(dispatch_size);
        // stream << synchronize();
        _data = std::move(t);
    }

    void load(BufferBase &&other) {
        _data = std::move(other);
    }

    void load(luisa::compute::Device &device, luisa::compute::Stream &stream, BufferBase &other) {
        _data = device.create_buffer<BaseType>(other.size());
        stream << _data.copy_from(other);
    }

    void load(luisa::compute::Device &device, luisa::compute::Stream &stream, void *data, size_t size, bool expand=false) {
        if (expand && _data.size_bytes() < size) {
            _data = device.create_buffer<BaseType>(size / stride);
        } else {
            LUISA_ASSERT(_data.size_bytes() >= size, "unexpand and not enough data space.");
        }
        stream << _data.copy_from(data);
    }

    template <class T>
    void load(luisa::compute::Device &device, luisa::compute::Stream &stream, luisa::vector<T> data, bool expand=true) {
        load(device, stream, data.data(), data.size_bytes(), expand);
    }
};