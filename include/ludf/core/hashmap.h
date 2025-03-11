#pragma once
#include <luisa/luisa-compute.h>
#include <luisa/dsl/syntax.h>
#include <ludf/util/kernel.h>


// only support uint and int key, value is uint
template <class Key>
struct Hashmap {
    BufferIndex _data;
    luisa::compute::Buffer<Key> _key;
    BufferIndex _counter;

    uint _capacity;

    void init(luisa::compute::Device &device, luisa::compute::Stream &stream, uint capacity, auto &&shader) {
        _capacity = capacity;

        _data = device.create_buffer<uint>(capacity);
        _counter = device.create_buffer<uint>(capacity);
        _key = device.create_buffer<Key>(capacity);

        stream << shader(_data, UINT_NULL).dispatch(capacity);
        stream << shader(_counter, 0u).dispatch(capacity);
    }
};

#define TEMPLATE_T()  \
    template<class T>

LUISA_BINDING_GROUP_TEMPLATE(TEMPLATE_T, Hashmap<T>, _data, _key, _counter, _capacity) {
    [[nodiscard]] luisa::compute::UInt hash(const luisa::compute::Var<T> &key) noexcept {
        auto uint_key = cast<uint>(key);
        uint_key ^= uint_key >> 16;
        uint_key *= 0x85ebca6b;
        uint_key ^= uint_key >> 13;
        uint_key *= 0xc2b2ae35;
        uint_key ^= uint_key >> 16;
        return uint_key % this->_capacity;
    }

    void insert(const luisa::compute::Var<T> &key, const luisa::compute::UInt &value) {
        auto idx = hash(key);
        auto count = this->_counter->atomic(idx).fetch_add(1u);
        $loop {
            auto old_value = this->_data.atomic(idx).compare_exchange(UINT_NULL, value);
            $if (old_value == UINT_NULL) {
                this->_key.write(idx, key);
                $break;
            };
            idx = (idx + 1u) % this->_capacity; 
        };
    }

    luisa::compute::UInt find(const luisa::compute::Var<T> &key) {
        auto idx = hash(key);
        return this->_data->read(idx);
    }
};

// #define LUISA_MAKE_HASHMAP_BINDING_GROUP(HashmapT) \
// LUISA_BINDING_GROUP(HashmapT, _data, _buffer, _capacity) {\
//     [[nodiscard]] luisa::compute::UInt hash(const auto &key) noexcept {\
//         return cast<uint>(key % this->_capacity);\
//     }\
// \
//     void insert(const auto &key, const luisa::compute::UInt &value) {\
//         auto idx = hash(key);\
//         auto data = this->_data->atomic(idx).compare_exchange(UINT_NULL, value);\
//     }\
// \
//     luisa::compute::UInt find(const auto &key) {\
//         auto idx = hash(key);\
//         return this->_data->read(idx);\
//     }\
// };

// using HashmapInt = Hashmap<int>;
// using HashmapUInt = Hashmap<uint>;

// LUISA_MAKE_HASHMAP_BINDING_GROUP(HashmapInt)
// LUISA_MAKE_HASHMAP_BINDING_GROUP(HashmapUInt)
