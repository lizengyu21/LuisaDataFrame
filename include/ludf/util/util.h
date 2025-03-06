#pragma once
#include <luisa/luisa-compute.h>
#include <ludf/column/column.h>
#include <ludf/util/kernel.h>

using namespace luisa;
using namespace luisa::compute;


template<class T>
inline void print_buffer(luisa::compute::Stream &stream, const luisa::compute::BufferView<T> & buffer);

template <class T>
BufferBase inverse_reindex(Device &device, Stream &stream, const BufferView<T> &data, const BufferViewIndex &indices, size_t res_size = 0) {
    LUISA_ASSERT(indices.size() <= data.size(), "indices' length must be less than data's.");
    LUISA_ASSERT(indices.size() > 0 && data.size() > 0, "invoke reindex must be non-empty.");
    BufferBase result;
    if (res_size == 0) result = device.create_buffer<BaseType>((indices.size() * sizeof(T)) / sizeof(BaseType));
    else result = device.create_buffer<BaseType>((res_size * sizeof(T)) / sizeof(BaseType));
    stream << ShaderCollector<T>::get_instance(device)->inverse_reindex_shader(result, data, indices).dispatch(indices.size());
    return std::move(result);
}

template <class T>
BufferBase reindex(Device &device, Stream &stream, const BufferView<T> &data, const BufferViewIndex &indices, size_t res_size = 0) {
    LUISA_ASSERT(indices.size() <= data.size(), "indices' length must be less than data's.");
    LUISA_ASSERT(indices.size() > 0 && data.size() > 0, "invoke reindex must be non-empty.");
    BufferBase result;
    if (res_size == 0) result = device.create_buffer<BaseType>((indices.size() * sizeof(T)) / sizeof(BaseType));
    else result = device.create_buffer<BaseType>((res_size * sizeof(T)) / sizeof(BaseType));
    stream << ShaderCollector<T>::get_instance(device)->reindex_shader(result, data, indices).dispatch(indices.size());
    return std::move(result);
}


struct concat_column {
    template <class T>
    void operator()(Device &device, Stream &stream, Column &lhs, Column &rhs) {
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
    BufferIndex operator()(Device &device, Stream &stream, Column &data, const FilterOp op, void *threshold) {
        BufferIndex indices = device.create_buffer<uint>(data.size());
        BufferIndex counter = device.create_buffer<uint>(1);
        stream << ShaderCollector<T>::get_instance(device)->reset_shader(counter).dispatch(1);
        T thres = *reinterpret_cast<T*>(threshold);
        std::cout << "::: " << thres << std::endl;
        stream << ShaderCollector<T>::get_instance(device)->make_inverse_reindex_shader_map[op](indices, counter, data.view<T>(), *reinterpret_cast<T*>(threshold)).dispatch(data.size());
        uint count;
        stream << counter.copy_to(&count) << synchronize();
        BufferIndex res = device.create_buffer<uint>(count);
        stream << ShaderCollector<uint>::get_instance(device)->copy_shader(res, indices).dispatch(count);
        return std::move(res);
    }
};

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



template<class T>
inline void print_buffer(luisa::compute::Stream &stream, const luisa::compute::BufferView<T> & buffer) {
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