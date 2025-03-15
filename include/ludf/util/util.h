#pragma once
#include <luisa/luisa-compute.h>
#include <ludf/column/column.h>
#include <ludf/util/kernel.h>
#include <any>
#include <luisa/backends/ext/cuda/lcub/device_radix_sort.h>
#include <ludf/core/hashmap.h>

template<class T>
inline void print_buffer(luisa::compute::Stream &stream, const luisa::compute::BufferView<T> & buffer);

BufferIndex inclusive_sum(luisa::compute::Device &device, luisa::compute::Stream &stream, BufferIndex &adjacent_diff_result);
BufferIndex exclusive_sum(luisa::compute::Device &device, luisa::compute::Stream &stream, BufferIndex &adjacent_diff_result);
BufferBase unique_count(luisa::compute::Device &device, luisa::compute::Stream &stream, BufferIndex &adjacent_diff_result, BufferIndex &indices, uint num_group);

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

        if (data._null_mask._data.size() != 0) {
            Bitmap null_mask;
            null_mask.init_zero(device, stream, indices.size(), ShaderCollector<uint>::get_instance(device)->set_shader);
            stream << ShaderCollector<T>::get_instance(device)->inverse_reindex_bitmap_shader(null_mask, data._null_mask, indices).dispatch(indices.size());
            data._null_mask = std::move(null_mask);
        }
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
        if (data._null_mask._data.size() == 0) {
            data._null_mask.init_zero(device, stream, data.size(), ShaderCollector<uint>::get_instance(device)->set_shader);
        }
        stream << ShaderCollector<T>::get_instance(device)->filter_reindex_shader(dst_view, src_view, data._null_mask, indices).dispatch(indices.size());
        data.load(std::move(res_buf));
        data._null_mask.init_zero(device, stream, data.size(), ShaderCollector<uint>::get_instance(device)->set_shader);
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

        if (data._null_mask._data.size() == 0) data._null_mask.init_zero(device, stream, data.size(), ShaderCollector<uint>::get_instance(device)->set_shader);

        stream << ShaderCollector<T>::get_instance(device)->make_inverse_reindex_shader_map[op](indices, counter, data.view<T>(), data._null_mask, thres).dispatch(data.size());
        uint count;
        stream << counter.copy_to(&count) << synchronize();
        if (count == 0) {
            return BufferIndex();
        }
        BufferIndex res = device.create_buffer<uint>(count);
        stream << ShaderCollector<uint>::get_instance(device)->copy_shader(res, indices).dispatch(count);
        // print_buffer(stream, res.view());
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

        if (data._null_mask._data.size() != 0) {
            if (order == SortOrder::Ascending) stream << ShaderCollector<T>::get_instance(device)->filter_set_shader(data_in_view, data._null_mask, std::numeric_limits<T>::max()).dispatch(num_item);
            else stream << ShaderCollector<T>::get_instance(device)->filter_set_shader(data_in_view, data._null_mask, std::numeric_limits<T>::lowest()).dispatch(num_item);
        }

        Buffer<int> temp_storage;
        size_t temp_storage_size = -1;

        if (order == SortOrder::Ascending) DeviceRadixSort::SortPairs(temp_storage_size, data_in_view, data_out.view().as<T>(), indices_in.view(), indices_out.view(), num_item);
        else DeviceRadixSort::SortPairsDescending(temp_storage_size, data_in_view, data_out.view().as<T>(), indices_in.view(), indices_out.view(), num_item);

        temp_storage = device.create_buffer<int>(temp_storage_size);
        if (order == SortOrder::Ascending) stream << DeviceRadixSort::SortPairs(temp_storage, data_in_view, data_out.view().as<T>(), indices_in.view(), indices_out.view(), num_item);
        else stream << DeviceRadixSort::SortPairsDescending(temp_storage, data_in_view, data_out.view().as<T>(), indices_in.view(), indices_out.view(), num_item);
        
        sorted_result.load(std::move(data_out));

        if (data._null_mask._data.size() != 0) {
            Bitmap null_mask;
            null_mask.init_zero(device, stream, num_item, ShaderCollector<uint>::get_instance(device)->set_shader);
            stream << ShaderCollector<T>::get_instance(device)->inverse_reindex_bitmap_shader(null_mask, data._null_mask, indices_out).dispatch(indices_out.size());
            sorted_result._null_mask = std::move(null_mask);
        }

        return std::move(indices_out);
    }
};

struct adjacent_diff {
    template <class T>
    BufferIndex operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &data) {
        
        using namespace luisa;
        using namespace luisa::compute;

        BufferView<T> data_view = data.view<T>();
        if (data._null_mask._data.size() == 0) {
            Bitmap null_mask;
            null_mask.init_zero(device, stream, data.size(), ShaderCollector<uint>::get_instance(device)->set_shader);
            data._null_mask = std::move(null_mask);
        }
        BufferIndex adjacent_diff_result = device.create_buffer<uint>(data.size());
        stream << ShaderCollector<uint>::get_instance(device)->reset_shader(adjacent_diff_result).dispatch(1);
        if (data.size() > 1) stream << ShaderCollector<T>::get_instance(device)->adjacent_diff_shader(data_view, data._null_mask, adjacent_diff_result).dispatch(data.size() - 1);
        return std::move(adjacent_diff_result);
    }
};

struct aggregate_column {
    template <class T>
    Column operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &data, const AggeragateOp &op, BufferIndex &indices, uint num_group) {
        using namespace luisa;
        using namespace luisa::compute;


        BufferView<T> data_view = data.view<T>();
        if (data._null_mask._data.size() == 0) {
            data._null_mask.init_zero(device, stream, data.size(), ShaderCollector<uint>::get_instance(device)->set_shader);
        }

        if (op == AggeragateOp::COUNT) {
            return Column{TypeId::UINT32};
        } else {
            BufferBase res_buf = device.create_buffer<BaseType>(num_group * sizeof(T) / sizeof(BaseType));
            Bitmap null_mask;
            null_mask.init_zero(device, stream, num_group, ShaderCollector<uint>::get_instance(device)->set_shader);
            // stream << ShaderCollector<uint>::get_instance(device)->reindex_bitmap_with_null_shader(null_mask, data._null_mask, inclusive_sum_result).dispatch(current_col.size());
            
            T init_value = op == AggeragateOp::MAX ? std::numeric_limits<T>::lowest() : 
                           op == AggeragateOp::MIN ? std::numeric_limits<T>::max() : static_cast<T>(0);

            stream << ShaderCollector<T>::get_instance(device)->set_shader(res_buf.view().as<T>(), init_value).dispatch(num_group);
            stream << ShaderCollector<T>::get_instance(device)->aggregate_shader_map[op](data_view, data._null_mask, res_buf.view().as<T>(), null_mask, indices, indices.size(), init_value).dispatch(indices.size());
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
        if (sum_data._null_mask._data.size() == 0) {
            sum_data._null_mask.init_zero(device, stream, sum_data.size(), ShaderCollector<uint>::get_instance(device)->set_shader);
        }
        BufferBase result = device.create_buffer<BaseType>(sum_data.size() * sizeof(float) / sizeof(BaseType));

        stream << ShaderCollector<T>::get_instance(device)->sum_to_mean_shader(data_view, sum_data._null_mask, count_data, result.view().as<float>()).dispatch(sum_data.size());

        sum_data.set_dtype(TypeId::FLOAT32);
        sum_data.load(std::move(result));
    }
};

struct apply_on_column_T {
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

struct apply_on_column_Ret_T {
    template <class Ret, class T>
    void operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &col, void *apply_func_ptr) {
        using namespace luisa;
        using namespace luisa::compute;

        BufferView<T> data_view = col.view<T>();
        BufferBase result = device.create_buffer<BaseType>(col.size() * sizeof(Ret) / sizeof(BaseType));
        auto shader = ShaderCollector<T>::get_instance(device)->template create_apply_Ret_T_shader<Ret>(device, apply_func_ptr);
        stream << shader(result.view().as<Ret>(), data_view).dispatch(col.size());

        col.load(std::move(result));
    }
};

struct outer_join {
    template <class T>
    std::pair<BufferIndex, BufferIndex> operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &left, Column &right) {
        using namespace luisa;
        using namespace luisa::compute;
        // std::cout << "LEFT JOIN\n";


        auto left_data_view = left.view<T>();
        auto right_data_view = right.view<T>();
        auto buffer_size = left.size() + right.size();

        BufferIndex join_count = device.create_buffer<uint>(buffer_size);
        Bitmap match_mask;
        match_mask.init_zero(device, stream, buffer_size, ShaderCollector<uint>::get_instance(device)->set_shader);

        stream << ShaderCollector<uint>::get_instance(device)->set_shader(join_count, 0u).dispatch(buffer_size);
        if (left._null_mask._data.size() == 0) left._null_mask.init_zero(device, stream, left.size(), ShaderCollector<uint>::get_instance(device)->set_shader);
        if (right._null_mask._data.size() == 0) right._null_mask.init_zero(device, stream, right.size(), ShaderCollector<uint>::get_instance(device)->set_shader);


        stream << ShaderCollector<T>::get_instance(device)->outer_join_count_shader(left_data_view, right_data_view, join_count, left._null_mask, right._null_mask, match_mask, left.size()).dispatch(luisa::compute::make_uint2(left.size(), right.size()));

        Buffer<uint> total_rows_buffer = device.create_buffer<uint>(1);
        uint total_rows; 
        stream << ShaderCollector<uint>::get_instance(device)->set_shader(total_rows_buffer, 0u).dispatch(1)
               << ShaderCollector<uint>::get_instance(device)->outer_sum_shader(join_count, left._null_mask, right._null_mask, match_mask, total_rows_buffer, left.size()).dispatch(buffer_size)
               << total_rows_buffer.copy_to(&total_rows) << synchronize();


        if (total_rows == 0) {
            return std::make_pair(Buffer<uint>{}, Buffer<uint>{});
        }
        // outer sum 左边的列将0的位置设置为1，右边的将没有匹配到（!match_mask->test(y)）的设置为1，则可得到整体的数量，并且后面可以直接用exclusive求index

        Buffer<uint> result_left = device.create_buffer<uint>(total_rows);
        Buffer<uint> result_right = device.create_buffer<uint>(total_rows);
        auto result_index_start = exclusive_sum(device, stream, join_count);

        BufferIndex slot_pointer = device.create_buffer<uint>(left.size());
        stream << ShaderCollector<uint>::get_instance(device)->set_shader(slot_pointer, 0u).dispatch(left.size());



        stream << ShaderCollector<T>::get_instance(device)->join_reindex_shader(
            left_data_view, right_data_view, left._null_mask, right._null_mask, result_index_start, slot_pointer, result_left, result_right).dispatch(luisa::compute::make_uint2(left.size(), right.size()));

        stream << ShaderCollector<T>::get_instance(device)->outer_join_match_mask_filter_shader(match_mask, left._null_mask, right._null_mask, result_index_start, result_left, result_right, left.size()).dispatch(buffer_size);

        return std::make_pair(std::move(result_left), std::move(result_right));
    }
};

struct inner_join {
    template <class T>
    std::pair<BufferIndex, BufferIndex> operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &left, Column &right) {
        using namespace luisa;
        using namespace luisa::compute;
        // std::cout << "LEFT JOIN\n";


        auto left_data_view = left.view<T>();
        auto right_data_view = right.view<T>();
        BufferIndex join_count = device.create_buffer<uint>(left.size());
        Bitmap match_mask;
        match_mask.init_zero(device, stream, left.size(), ShaderCollector<uint>::get_instance(device)->set_shader);

        stream << ShaderCollector<uint>::get_instance(device)->set_shader(join_count, 0u).dispatch(left.size());
        if (left._null_mask._data.size() == 0) left._null_mask.init_zero(device, stream, left.size(), ShaderCollector<uint>::get_instance(device)->set_shader);
        if (right._null_mask._data.size() == 0) right._null_mask.init_zero(device, stream, right.size(), ShaderCollector<uint>::get_instance(device)->set_shader);


        stream << ShaderCollector<T>::get_instance(device)->join_count_shader(left_data_view, right_data_view, join_count, left._null_mask, right._null_mask, match_mask).dispatch(luisa::compute::make_uint2(left.size(), right.size()));


        Buffer<uint> total_rows_buffer = device.create_buffer<uint>(1);
        uint total_rows; 
        stream << ShaderCollector<uint>::get_instance(device)->set_shader(total_rows_buffer, 0u).dispatch(1)
               << ShaderCollector<uint>::get_instance(device)->inner_sum_shader(join_count, total_rows_buffer).dispatch(left.size())
               << total_rows_buffer.copy_to(&total_rows) << synchronize();

        if (total_rows == 0) {
            return std::make_pair(Buffer<uint>{}, Buffer<uint>{});
        }

        Buffer<uint> result_left = device.create_buffer<uint>(total_rows);
        Buffer<uint> result_right = device.create_buffer<uint>(total_rows);
        auto result_index_start = exclusive_sum(device, stream, join_count);
        BufferIndex slot_pointer = device.create_buffer<uint>(left.size());
        stream << ShaderCollector<uint>::get_instance(device)->set_shader(slot_pointer, 0u).dispatch(left.size());



        stream << ShaderCollector<T>::get_instance(device)->join_reindex_shader(
            left_data_view, right_data_view, left._null_mask, right._null_mask, result_index_start, slot_pointer, result_left, result_right).dispatch(luisa::compute::make_uint2(left.size(), right.size()));

        return std::make_pair(std::move(result_left), std::move(result_right));
    }
};


struct left_join {
    template <class T>
    std::pair<BufferIndex, BufferIndex> operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &left, Column &right) {
        using namespace luisa;
        using namespace luisa::compute;
        // std::cout << "LEFT JOIN\n";


        auto left_data_view = left.view<T>();
        auto right_data_view = right.view<T>();
        BufferIndex join_count = device.create_buffer<uint>(left.size());
        Bitmap match_mask;
        match_mask.init_zero(device, stream, left.size(), ShaderCollector<uint>::get_instance(device)->set_shader);

        stream << ShaderCollector<uint>::get_instance(device)->set_shader(join_count, 0u).dispatch(left.size());
        if (left._null_mask._data.size() == 0) left._null_mask.init_zero(device, stream, left.size(), ShaderCollector<uint>::get_instance(device)->set_shader);
        if (right._null_mask._data.size() == 0) right._null_mask.init_zero(device, stream, right.size(), ShaderCollector<uint>::get_instance(device)->set_shader);


        stream << ShaderCollector<T>::get_instance(device)->join_count_shader(left_data_view, right_data_view, join_count, left._null_mask, right._null_mask, match_mask).dispatch(luisa::compute::make_uint2(left.size(), right.size()));

        // print_buffer(stream, join_count.view());

        Buffer<uint> total_rows_buffer = device.create_buffer<uint>(1);
        uint total_rows; 
        stream << ShaderCollector<uint>::get_instance(device)->set_shader(total_rows_buffer, 0u).dispatch(1)
               << ShaderCollector<uint>::get_instance(device)->left_sum_shader(join_count, left._null_mask, total_rows_buffer).dispatch(left.size())
               << total_rows_buffer.copy_to(&total_rows) << synchronize();
        
        if (total_rows == 0) {
            return std::make_pair(Buffer<uint>{}, Buffer<uint>{});
        }
        // print_buffer(stream, join_count.view());
        // print_buffer(stream, total_rows_buffer.view());

        Buffer<uint> result_left = device.create_buffer<uint>(total_rows);
        Buffer<uint> result_right = device.create_buffer<uint>(total_rows);
        auto result_index_start = exclusive_sum(device, stream, join_count);
        BufferIndex slot_pointer = device.create_buffer<uint>(left.size());
        stream << ShaderCollector<uint>::get_instance(device)->set_shader(slot_pointer, 0u).dispatch(left.size());

        // print_buffer(stream, result_index_start.view());

        stream << ShaderCollector<T>::get_instance(device)->join_reindex_shader(
            left_data_view, right_data_view, left._null_mask, right._null_mask, result_index_start, slot_pointer, result_left, result_right).dispatch(luisa::compute::make_uint2(left.size(), right.size()));

        stream << ShaderCollector<T>::get_instance(device)->join_match_mask_filter_shader(match_mask, left._null_mask, result_index_start, result_left, result_right).dispatch(left.size());

        return std::make_pair(std::move(result_left), std::move(result_right));
    }
};

struct right_join {
    template <class T>
    std::pair<BufferIndex, BufferIndex> operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, Column &left, Column &right) {
        using namespace luisa;
        using namespace luisa::compute;
        BufferIndex l;
        BufferIndex r;
        std::tie(r, l) = left_join{}.operator()<T>(device, stream, right, left);
        return std::make_pair(std::move(l), std::move(r));
    }
};

struct join_reindex_col {
    template <class T>
    Column operator()(luisa::compute::Device &device, luisa::compute::Stream &stream, BufferIndex &indices, Column &data) {
        using namespace luisa;
        using namespace luisa::compute;
        BufferView<T> data_view = data.view<T>();
        if (indices.size() == 0) {
            return Column{data.dtype()};
        }
        BufferBase res_buf = device.create_buffer<BaseType>(indices.size() * sizeof(T) / sizeof(BaseType));
        Bitmap null_mask;
        null_mask.init_zero(device, stream, indices.size(), ShaderCollector<uint>::get_instance(device)->set_shader);
        if (data._null_mask._data.size() == 0) data._null_mask.init_zero(device, stream, data.size(), ShaderCollector<uint>::get_instance(device)->set_shader);
        
        stream << ShaderCollector<T>::get_instance(device)->reindex_with_nullmask_shader(res_buf.view().as<T>(), data_view, indices, data._null_mask, null_mask).dispatch(indices.size());
        return Column{std::move(res_buf), std::move(null_mask), data.dtype()};
    }
};


void inline fill_join_result(luisa::compute::Device &device, luisa::compute::Stream &stream, BufferIndex &indices, luisa::unordered_map<luisa::string, Column> &data, luisa::unordered_map<luisa::string, Column> &result) {
    for (auto it = data.begin(); it != data.end(); ++it) {
        auto name = it->first;
        if (result.find(name) != result.end()) {
            LUISA_WARNING("JOIN INTERUPT: Join two table should NOT have the same column name [{}]", name);
            return;
        }
        auto res_col = type_dispatcher(it->second.dtype().id(), join_reindex_col{}, device, stream, indices, it->second);
        result.insert({name, std::move(res_col)});
    }
}

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
    
    auto max_len = 40;
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