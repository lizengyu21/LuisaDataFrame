#pragma once
#include <luisa/luisa-compute.h>
#include <iostream>


using BufferViewIndex = luisa::compute::BufferView<uint>;
using BufferIndex = luisa::compute::Buffer<uint>;
using BaseType = int;
using BufferBase = luisa::compute::Buffer<BaseType>;

// namespace lc = luisa::compute;

enum class TypeId {
    EMPTY = 0,
    INT32,
    UINT32,
    FLOAT32,
    TIMESTAMP,  // INT32,
    // INT64,
};

enum class AggeragateOp {
    SUM = 0,
    COUNT,
    MAX,
    MIN,
    MEAN,
};

enum class FilterOp {
    LESS = 0,
    LESS_EQUAL,
    GREATER,
    GREATER_EQUAL,
    EQUAL,
    NOT_EQUAL,
};

enum class SortOrder {
    Ascending = 0,
    Descending,
};

enum class JoinType {
    INNER = 0,
    LEFT,
    RIGHT,
    OUTER,
};

class DataType {
public:
    TypeId _id;
    constexpr DataType(TypeId id = TypeId::EMPTY) : _id(id) {}
    DataType(const DataType &) = default;
    DataType(DataType &&) = default;

    DataType &operator=(const DataType &) = default;
    DataType &operator=(DataType &&) = default;

    [[nodiscard]] constexpr TypeId id() const noexcept { return _id; }
};

inline constexpr bool operator==(const DataType &l, const DataType &r) {
    return l.id() == r.id();
}

inline constexpr bool operator!=(const DataType &l, const DataType &r) {
    return !(l == r);
}
