#pragma once
#include <luisa/luisa-compute.h>
#include <ludf/core/type.h>

template <class Key>
struct Hashmap {
    BufferBase _key;
    BufferIndex _counter;
    BufferIndex _offset;
    uint _capacity;
};

#define TEMPLATE_T()  \
    template<class T>

LUISA_BINDING_GROUP_TEMPLATE(TEMPLATE_T, Hashmap<T>, _key, _counter, _offset, _capacity) {
    [[nodiscard]] luisa::compute::UInt hash(const luisa::compute::Var<T> &key) noexcept {
        using namespace luisa;
        using namespace luisa::compute;

        auto uint_key = cast<uint>(key);
        uint_key = uint_key * 0xcc9e2d51u;
        uint_key = (uint_key << 15u) | (uint_key >> (32u - 15u));
        uint_key = uint_key * 0x1b873593u;
        
        UInt hash = MURMURHASH_SEED;

        hash = hash ^ uint_key;
        hash = (hash << 13) | (hash >> (32u - 13));
        hash = hash * 5u + 0xe6546b64u;

        hash = hash ^ 4u; // 数据长度（假设输入为 4 字节）
        hash = hash ^ (hash >> 16u);
        hash = hash * 0x85ebca6bu;
        hash = hash ^ (hash >> 13u);
        hash = hash * 0xc2b2ae35u;
        hash = hash ^ (hash >> 16u);

        return hash % this->_capacity;
    }

    void insert(const luisa::compute::Var<T> &key) noexcept {
        using namespace luisa;
        using namespace luisa::compute;
        UInt slot = this->hash(key);
        $loop {
            auto cur_key = this->_key.read(slot);
            
            $break;
        };
    }
};
