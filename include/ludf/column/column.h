#pragma once
#include <ludf/core/type_dispatcher.h>
#include <luisa/luisa-compute.h>
#include <ludf/util/kernel.h>

using namespace luisa;
using namespace luisa::compute;

class Column {
    static const size_t stride = sizeof(BaseType);
public:
    DataType _dtype;
    Buffer<BaseType> _data;

    Column(DataType dtype = DataType{TypeId::EMPTY}) : _dtype(dtype) {}
    Column(BufferBase &&data, DataType dtype) : _dtype(dtype), _data(std::move(data)) {}

    template <class T>
    BufferView<T> view() {
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

    Column clone(Device &device, Stream &stream) const {
        BufferBase t = device.create_buffer<BaseType>(_data.size());
        stream << t.copy_from(_data);
        return std::move(Column{std::move(t), _dtype});
    }

    void resize(Device &device, Stream &stream, size_t size_byte) {
        auto t = device.create_buffer<BaseType>(size_byte / stride);
        size_t dispatch_size = t.size() < _data.size() ? t.size() : _data.size();
        stream << ShaderCollector<BaseType>::get_instance(device)->copy_shader(t, _data).dispatch(dispatch_size);
        // stream << synchronize();
        _data = std::move(t);
    }

    void load(Device &device, Stream &stream, void *data, size_t size, bool expand=false) {
        if (expand && _data.size_bytes() < size) {
            _data = device.create_buffer<BaseType>(size / stride);
        } else {
            LUISA_ASSERT(_data.size_bytes() >= size, "unexpand and not enough data space.");
        }
        stream << _data.copy_from(data);
    }

    template <class T>
    void load(Device &device, Stream &stream, vector<T> data, bool expand=true) {
        load(device, stream, data.data(), data.size_bytes(), expand);
    }
};