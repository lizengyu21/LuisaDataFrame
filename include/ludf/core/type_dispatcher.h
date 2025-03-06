#pragma once
#include <ludf/core/type.h>

template <class T>
inline constexpr TypeId type_to_id() {
    return TypeId::EMPTY;
}

template <TypeId id>
struct id_to_type_impl {
    using type = void;
};

template <TypeId id>
using id_to_type = typename id_to_type_impl<id>::type;

#define TYPE_MAPPING(TYPE, ID) \
    template <> \
    inline constexpr TypeId type_to_id<TYPE>() { \
        return ID; \
    } \
    template <> \
    struct id_to_type_impl<ID> { \
        using type = TYPE; \
    }

TYPE_MAPPING(int, TypeId::INT32);
TYPE_MAPPING(uint, TypeId::UINT32);
TYPE_MAPPING(float, TypeId::FLOAT32);

#undef TYPE_MAPPING

template <TypeId id>
inline constexpr size_t size_of() {
    return sizeof(id_to_type<id>);
};

#define SIZE_MAPPING(ID, SIZE) \
    case ID: { \
        return SIZE; \
    }

inline size_t id_to_size(TypeId id) {
    switch (id) {
        SIZE_MAPPING(TypeId::EMPTY, 0)
        SIZE_MAPPING(TypeId::INT32, 4)
        SIZE_MAPPING(TypeId::UINT32, 4)
        SIZE_MAPPING(TypeId::FLOAT32, 4)
    default:
        return -1;
        break;
    }
}

#undef SIZE_MAPPING

template <typename Functor, typename... Args>
decltype(auto) type_dispatcher(const DataType &dtype, Functor f, Args&&... args) {
    switch(dtype.id()) {
        case TypeId::FLOAT32: return f.template operator()<id_to_type<TypeId::FLOAT32>>(std::forward<Args>(args)...); break;
        case TypeId::INT32: return f.template operator()<id_to_type<TypeId::INT32>>(std::forward<Args>(args)...); break;
        case TypeId::UINT32: return f.template operator()<id_to_type<TypeId::UINT32>>(std::forward<Args>(args)...); break;
        // TYPEID_MAPPING(TypeId::FLOAT32);
        // TYPEID_MAPPING(TypeId::INT32);
        // TYPEID_MAPPING(TypeId::UINT32);
    }
}


