#pragma once
#include <ludf/core/type.h>

template <TypeId id>
struct id_to_type_impl {
    using type = void;
};

template <TypeId id>
using id_to_type = typename id_to_type_impl<id>::type;

template <typename T>
inline constexpr TypeId base_type_to_id() {
  return TypeId::EMPTY;
};

template <typename T>
constexpr inline TypeId type_to_id() {
  return base_type_to_id<std::remove_cv_t<T>>();
}

#define ID_TO_TYPE_MAPPING(TYPE, ID) \
    template <> \
    struct id_to_type_impl<ID> { \
        using type = TYPE; \
    }

ID_TO_TYPE_MAPPING(int32_t, TypeId::INT32);
ID_TO_TYPE_MAPPING(uint32_t, TypeId::UINT32);
ID_TO_TYPE_MAPPING(float, TypeId::FLOAT32);
ID_TO_TYPE_MAPPING(uint32_t, TypeId::TIMESTAMP);

#undef ID_TO_TYPE_MAPPING

#define TYPE_TO_ID_MAPPING(TYPE, ID) \
    template <> \
    constexpr inline TypeId base_type_to_id<TYPE>() { \
      return ID; \
    }

TYPE_TO_ID_MAPPING(int32_t, TypeId::INT32);
TYPE_TO_ID_MAPPING(uint32_t, TypeId::UINT32);
TYPE_TO_ID_MAPPING(float, TypeId::FLOAT32);

#undef TYPE_TO_ID_MAPPING

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

inline luisa::string type_id_string(const TypeId id) {
    switch (id) {
        case TypeId::EMPTY: return "NONE";
        case TypeId::INT32: return "INT32";
        case TypeId::UINT32: return "UINT32";
        case TypeId::FLOAT32: return "FLOAT32";
        case TypeId::TIMESTAMP: return "TIMESTAMP";
        default: return "UNKNOWN";
    }
}

template <class T>
bool same_type(const TypeId &id) {
    return false;
}

template <>
bool same_type<int32_t>(const TypeId &id) {
    return id == TypeId::INT32;
}

template <>
bool same_type<uint32_t>(const TypeId &id) {
    return id == TypeId::UINT32 || id == TypeId::TIMESTAMP;
}

template <>
bool same_type<float>(const TypeId &id) {
    return id == TypeId::FLOAT32;
}


