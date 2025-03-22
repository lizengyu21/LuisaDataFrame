#pragma once
#include <luisa/luisa-compute.h>
#include <ludf/core/type.h>

template <class Key>
struct Hashmap {
    BufferIndex _key;
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

        auto uint_key = as<uint>(key);
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
        // device_log("Insert Key {}", key);
        UInt slot = this->hash(key);
        UInt uint_key = as<uint>(key);

        $while (true) {
            auto cur_key = this->_key.read(slot);
            $if (cur_key == uint_key) {
                // 找到了相同的 key
                this->_counter.atomic(slot).fetch_add(1u);
                $break;
            } $elif (cur_key == UINT_NULL) {
                // 找到了一个空的 slot，尝试插入
                $if (this->_key.atomic(slot).compare_exchange(UINT_NULL, uint_key) == UINT_NULL) {
                    // 成功分配了一个 slot
                    this->_counter.atomic(slot).fetch_add(1u);
                    $break;
                } $else {
                    $continue;
                };
            };
            slot = (slot + 1u) % this->_capacity;
        };
    }

    [[nodiscard]] luisa::compute::UInt find(const luisa::compute::Var<T> &key) noexcept {
        using namespace luisa;
        using namespace luisa::compute;
        UInt slot = this->hash(key);
        UInt uint_key = as<uint>(key);
        UInt result;
        $while (true) {
            auto cur_key = this->_key.read(slot);
            $if (cur_key == uint_key) {
                result = this->_counter.read(slot);
                $break;
            } $elif (cur_key == UINT_NULL) {
                result = 0u;
                $break;
            };
            slot = (slot + 1u) % this->_capacity;
        };
        return result;
    }
};
