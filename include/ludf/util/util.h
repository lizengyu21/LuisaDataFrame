#pragma once
#include <luisa/luisa-compute.h>
#include <ludf/column/column.h>
#include <ludf/util/kernel.h>
#include <any>
#include <luisa/backends/ext/cuda/lcub/device_radix_sort.h>


template<class T>
inline void print_buffer(luisa::compute::Stream &stream, const luisa::compute::BufferView<T> & buffer);

// template <class T>
// BufferBase inverse_reindex(Device &device, Stream &stream, const BufferView<T> &data, const BufferViewIndex &indices, size_t res_size = 0) {
//     LUISA_ASSERT(indices.size() <= data.size(), "indices' length must be less than data's.");
//     LUISA_ASSERT(indices.size() > 0 && data.size() > 0, "invoke reindex must be non-empty.");
//     BufferBase result;
//     if (res_size == 0) result = device.create_buffer<BaseType>((indices.size() * sizeof(T)) / sizeof(BaseType));
//     else result = device.create_buffer<BaseType>((res_size * sizeof(T)) / sizeof(BaseType));
//     stream << ShaderCollector<T>::get_instance(device)->inverse_reindex_shader(result, data, indices).dispatch(indices.size());
//     return std::move(result);
// }

// template <class T>
// BufferBase reindex(Device &device, Stream &stream, const BufferView<T> &data, const BufferViewIndex &indices, size_t res_size = 0) {
//     LUISA_ASSERT(indices.size() <= data.size(), "indices' length must be less than data's.");
//     LUISA_ASSERT(indices.size() > 0 && data.size() > 0, "invoke reindex must be non-empty.");
//     BufferBase result;
//     if (res_size == 0) result = device.create_buffer<BaseType>((indices.size() * sizeof(T)) / sizeof(BaseType));
//     else result = device.create_buffer<BaseType>((res_size * sizeof(T)) / sizeof(BaseType));
//     stream << ShaderCollector<T>::get_instance(device)->reindex_shader(result, data, indices).dispatch(indices.size());
//     return std::move(result);
// }

struct inverse_reindex {
    template <class T>
    void operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &data, BufferIndex &indices) {
        using namespace luisa;
        using namespace luisa::compute;

        if (indices.size() == 0) {
            data.resize(device, stream, 0);
            return;
        }
        BufferBase res_buf = device.create_buffer<BaseType>(indices.size() * sizeof(T) / sizeof(BaseType));
        auto dst_view = res_buf.view().as<T>();
        auto src_view = data.view<T>();
        stream << ShaderCollector<T>::get_instance(device)->inverse_reindex_shader(dst_view, src_view, indices).dispatch(indices.size());
        data.load(std::move(res_buf));
    }
};

struct reindex {
    template <class T>
    void operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &data, BufferIndex &indices, uint res_size = 0) {
        using namespace luisa;
        using namespace luisa::compute;

        if (indices.size() == 0) {
            data.resize(device, stream, 0);
            return;
        }

        BufferBase res_buf;
        if (res_size > 0) res_buf = device.create_buffer<BaseType>(res_size * sizeof(T) / sizeof(BaseType));
        else res_buf = device.create_buffer<BaseType>(indices.size() * sizeof(T) / sizeof(BaseType));

        auto dst_view = res_buf.view().as<T>();
        auto src_view = data.view<T>();
        stream << ShaderCollector<T>::get_instance(device)->reindex_shader(dst_view, src_view, indices).dispatch(indices.size());
        data.load(std::move(res_buf));
    }
};

struct concat_column {
    template <class T>
    void operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &lhs, Column &rhs) {
        using namespace luisa;
        using namespace luisa::compute;

        LUISA_ASSERT(lhs._dtype.id() == rhs._dtype.id(), "concat two col must be same type");
        auto start_id = lhs.size();
        lhs.resize(device, stream, rhs.size_bytes() + lhs.size_bytes());
        auto dst_view = lhs.view<T>().subview(start_id, rhs.size());
        auto src_view = rhs.view<T>();
        stream << ShaderCollector<T>::get_instance(device)->copy_shader(dst_view, src_view).dispatch(rhs.size());
    }
};

struct make_inverse_reindex {
    template <class T>
    BufferIndex operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &data, const FilterOp op, std::any threshold) { 
        using namespace luisa;
        using namespace luisa::compute;

        BufferIndex indices = device.create_buffer<uint>(data.size());
        BufferIndex counter = device.create_buffer<uint>(1);
        stream << ShaderCollector<id_to_type<TypeId::UINT32>>::get_instance(device)->reset_shader(counter).dispatch(1);
        T thres = std::any_cast<T>(threshold);
        stream << ShaderCollector<T>::get_instance(device)->make_inverse_reindex_shader_map[op](indices, counter, data.view<T>(), thres).dispatch(data.size());
        uint count;
        stream << counter.copy_to(&count) << synchronize();
        if (count == 0) {
            return BufferIndex();
        }
        BufferIndex res = device.create_buffer<uint>(count);
        stream << ShaderCollector<uint>::get_instance(device)->copy_shader(res, indices).dispatch(count);
        return std::move(res);
    }
};

struct sort_column {
    template <class T>
    BufferIndex operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &data, Column &sorted_result, const SortOrder &order = SortOrder::Ascending) {
        using namespace luisa;
        using namespace luisa::compute;
        using namespace luisa::compute::cuda::lcub;

        size_t num_item = data.size();
        BufferIndex indices_in = device.create_buffer<uint>(num_item);
        BufferIndex indices_out = device.create_buffer<uint>(num_item);
        BufferView<T> data_in_view = data.view<T>();
        BufferBase data_out = device.create_buffer<BaseType>(num_item * sizeof(T) / sizeof(BaseType));
        stream << ShaderCollector<uint>::get_instance(device)->arange_shader(indices_in).dispatch(num_item) << synchronize();

        Buffer<int> temp_storage;
        size_t temp_storage_size = -1;

        if (order == SortOrder::Ascending) DeviceRadixSort::SortPairs(temp_storage_size, data_in_view, data_out.view().as<T>(), indices_in.view(), indices_out.view(), num_item);
        else DeviceRadixSort::SortPairsDescending(temp_storage_size, data_in_view, data_out.view().as<T>(), indices_in.view(), indices_out.view(), num_item);

        temp_storage = device.create_buffer<int>(temp_storage_size);
        if (order == SortOrder::Ascending) stream << DeviceRadixSort::SortPairs(temp_storage, data_in_view, data_out.view().as<T>(), indices_in.view(), indices_out.view(), num_item);
        else stream << DeviceRadixSort::SortPairsDescending(temp_storage, data_in_view, data_out.view().as<T>(), indices_in.view(), indices_out.view(), num_item);
        
        sorted_result.load(std::move(data_out));

        return std::move(indices_out);
    }
};

struct adjacent_diff {
    template <class T>
    BufferIndex operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &data) {
        
        using namespace luisa;
        using namespace luisa::compute;

        BufferView<T> data_view = data.view<T>();

        BufferIndex adjacent_diff_result = device.create_buffer<uint>(data.size());
        stream << ShaderCollector<uint>::get_instance(device)->reset_shader(adjacent_diff_result).dispatch(1);
        if (data.size() > 1) stream << ShaderCollector<T>::get_instance(device)->adjacent_diff_shader(data_view, adjacent_diff_result).dispatch(data.size() - 1);
        return std::move(adjacent_diff_result);
    }
};

struct aggregate_column {
    template <class T>
    Column operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &data, const AggeragateOp &op, BufferIndex &adjacent_diff_result, BufferIndex &indices, uint num_group) {
        using namespace luisa;
        using namespace luisa::compute;


        BufferView<T> data_view = data.view<T>();

        if (op == AggeragateOp::COUNT) {
            return Column{TypeId::UINT32};
        } else {
            BufferBase res_buf = device.create_buffer<BaseType>(num_group * sizeof(T) / sizeof(BaseType));
            T init_value = op == AggeragateOp::MAX ? std::numeric_limits<T>::lowest() : 
                           op == AggeragateOp::MIN ? std::numeric_limits<T>::max() : static_cast<T>(0);
                           
            stream << ShaderCollector<T>::get_instance(device)->set_shader(res_buf.view().as<T>(), init_value).dispatch(num_group);
            stream << ShaderCollector<T>::get_instance(device)->aggregate_shader_map[op](data_view, res_buf.view().as<T>(), indices, indices.size(), init_value).dispatch(indices.size());
            return Column{std::move(res_buf), data.dtype()};
        }

        Column result{data.dtype()};

        return std::move(result);
    }
};

struct sum_to_mean {
    template <class T>
    void operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &sum_data, BufferViewIndex count_data) {
        
        using namespace luisa;
        using namespace luisa::compute;

        BufferView<T> data_view = sum_data.view<T>();
        BufferBase result = device.create_buffer<BaseType>(sum_data.size() * sizeof(float) / sizeof(BaseType));

        stream << ShaderCollector<T>::get_instance(device)->sum_to_mean_shader(data_view, count_data, result.view().as<float>()).dispatch(sum_data.size());

        sum_data.set_dtype(TypeId::FLOAT32);
        sum_data.load(std::move(result));
    }
};

struct apply_on_column {
    template <class T>
    void operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &col, void *apply_func_ptr) {
        using namespace luisa;
        using namespace luisa::compute;

        BufferView<T> data_view = col.view<T>();
        BufferBase result = device.create_buffer<BaseType>(col.size() * sizeof(T) / sizeof(BaseType));
        ShaderCollector<T>::get_instance(device)->create_apply_shader(device, apply_func_ptr);
        stream << ShaderCollector<T>::get_instance(device)->apply_shader(result.view().as<T>(), data_view).dispatch(col.size());

        col.load(std::move(result));

    }
};

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

BufferIndex inclusive_sum(luisa::compute::Device &device, luisa::compute::Stream &stream, BufferIndex &adjacent_diff_result);
BufferBase unique_count(luisa::compute::Device &device, luisa::compute::Stream &stream, BufferIndex &adjacent_diff_result, BufferIndex &indices, uint num_group);

// template <class T>
// BufferIndex make_filter_indices(Device &device, Stream &stream, const BufferView<T> &data, const FilterOp &op, const T &threshold) {
//     AtomicQueue<uint> queue(device);
//     queue.resize(device, data.size());
//     queue.reset(stream);
//     Clock clock;
//     clock.tic();
//     auto make_indices_shader = device.compile<1>([&](Var<T> thres){
//         auto x = dispatch_x();
//         queue.push_if(x > thres, x);
//     });
//     LUISA_INFO("{} ms", clock.toc());
//     // static auto
//     stream << make_indices_shader(threshold).dispatch(data.size());
//     uint count;
//     stream << queue._counter.copy_to(&count) << synchronize();
//     if (count == 0) {
//         return BufferIndex();
//     }
//     BufferIndex res = device.create_buffer<uint>(count);
//     static auto copy_shader = device.compile<1>([](BufferVar<uint> src, BufferVar<uint> dst){
//         auto x = dispatch_x();
//         dst.write(x, src.read(x));
//     });

//     stream << copy_shader(queue._buffer, res).dispatch(count);

//     return std::move(res);
// } 


struct print_column {
    template <class T>
    void operator()(luisa::compute::Stream &stream, Column &data) {
        if (data.size() == 0) {
            std::cout << "[]" << std::endl;
            return;
        }
        print_buffer(stream, data.view<T>());
    }
};

template<class T>
inline void print_buffer(luisa::compute::Stream &stream, const luisa::compute::BufferView<T> & buffer) {
    using namespace luisa;
    using namespace luisa::compute;
    
    auto max_len = 20;
    auto size = buffer.size();
    if (size == 0) {
        std::cout << "[]" << std::endl;
        return;
    }
    luisa::vector<T> host_data(size);
    if (size > max_len) {
        std::cout << "Total Length " << size << " <==> ";
    }
    stream << buffer.copy_to(host_data.data()) << synchronize();
    std::cout << '[';
    for (int i = 0; i < host_data.size() && i < max_len; ++i) {
        std::cout << host_data[i] << ", ";
    }
    if (size > max_len) {
        std::cout << "...]";
    } else {
        std::cout << "]";
    }
    
    std::cout << std::endl;
}