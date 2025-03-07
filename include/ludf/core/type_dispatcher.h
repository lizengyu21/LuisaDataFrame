#pragma once
#include <ludf/core/type.h>

template <TypeId id>
struct id_to_type_impl {
    using type = void;
};

template <TypeId id>
using id_to_type = typename id_to_type_impl<id>::type;

#define TYPE_MAPPING(TYPE, ID) \
    template <> \
    struct id_to_type_impl<ID> { \
        using type = TYPE; \
    }

TYPE_MAPPING(int32_t, TypeId::INT32);
TYPE_MAPPING(uint32_t, TypeId::UINT32);
TYPE_MAPPING(float, TypeId::FLOAT32);
TYPE_MAPPING(int32_t, TypeId::TIMESTAMP);

#undef TYPE_MAPPING

template <TypeId id>
inline constexpr size_t size_of() {
    return sizeof(id_to_type<id>);
};


#define SIZE_MAPPING(ID) \
case ID: { \
    return sizeof(id_to_type<ID>); \
}

inline size_t id_to_size(TypeId id) {
    switch (id) {
        // SIZE_MAPPING(TypeId::EMPTY, 0)
        case TypeId::EMPTY: return 0;
        SIZE_MAPPING(TypeId::INT32)
        SIZE_MAPPING(TypeId::UINT32)
        SIZE_MAPPING(TypeId::FLOAT32)
        SIZE_MAPPING(TypeId::TIMESTAMP)
        // SIZE_MAPPING(TypeId::INT64)
    default:
        return -1;
        break;
    }
}

#undef SIZE_MAPPING

template <typename Functor, typename... Args>
decltype(auto) type_dispatcher(const DataType &dtype, Functor f, Args&&... args) {
    switch(dtype.id()) {
        case TypeId::FLOAT32: return f.template operator()<id_to_type<TypeId::FLOAT32>>(std::forward<Args>(args)...);
        case TypeId::INT32: return f.template operator()<id_to_type<TypeId::INT32>>(std::forward<Args>(args)...);
        case TypeId::UINT32: return f.template operator()<id_to_type<TypeId::UINT32>>(std::forward<Args>(args)...);
        case TypeId::TIMESTAMP: return f.template operator()<id_to_type<TypeId::TIMESTAMP>>(std::forward<Args>(args)...);
    }
    LUISA_ERROR_WITH_LOCATION("A mismatched functor is invoked");
    return f.template operator()<id_to_type<TypeId::INT32>>(std::forward<Args>(args)...);
}

inline luisa::string agg_op_string(const AggeragateOp op) {
    switch (op) {
    case AggeragateOp::SUM: return "SUM";
    case AggeragateOp::MIN: return "MIN";
    case AggeragateOp::MAX: return "MAX";
    case AggeragateOp::COUNT: return "COUNT";
    case AggeragateOp::MEAN: return "MEAN";
    default: return "UNKNOWN";
    }
}


