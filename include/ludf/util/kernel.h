#pragma once
#include <luisa/luisa-compute.h>
#include <ludf/core/type.h>

using namespace luisa;
using namespace luisa::compute;

template <class T>
class ShaderCollector {
private:
    ShaderCollector() = delete;

    void create_make_reindex_shader(Device &device, const FilterOp &op, auto &&filter) {
        if (make_inverse_reindex_shader_map.find(op) != make_inverse_reindex_shader_map.end()) return;

        auto shader = device.compile<1>([&](BufferUInt indices, BufferUInt counter, BufferVar<T> data, Var<T> threshold){
            auto x = dispatch_x(); 
            auto pred = filter(data.read(x), threshold);
            Shared<uint> index{1}; 
            $if (thread_x() == 0u) { index.write(0u, 0u); }; 
            sync_block(); 
            auto local_index = def(0u); 
            $if (pred) { local_index = index.atomic(0).fetch_add(1u); }; 
            sync_block(); 
            $if (thread_x() == 0u) { 
                auto local_count = index.read(0u); 
                auto global_offset = counter->atomic(0u).fetch_add(local_count); 
                index.write(0u, global_offset); 
            }; 
            sync_block(); 
            $if (pred) { 
                auto global_index = index.read(0u) + local_index; 
                indices->write(global_index, x); 
            }; 
        });

        make_inverse_reindex_shader_map[op] = std::move(shader);
    }

    ShaderCollector(luisa::compute::Device &device) {
        reset_shader = device.compile<1>([](BufferVar<uint> counter) {
            counter.write(0, 0u);
        });
        copy_shader = device.compile<1>([](BufferVar<T> dst, BufferVar<T> src) {
            auto x = dispatch_x();
            dst.write(x, src.read(x));
        });
        reindex_shader = device.compile<1>([](BufferVar<T> dst, BufferVar<T> src, BufferVar<uint> idx){
            // dst[idx[x]] = src[x];
            // src:   [x]
            //         |
            //         "
            // dst: idx[x]
            auto x = dispatch_x();
            dst.write(idx->read(x), src.read(x));
        });
        inverse_reindex_shader = device.compile<1>([](BufferVar<T> dst, BufferVar<T> src, BufferVar<uint> idx){
            // dst[x] = src[idx[x]];
            // src: idx[x]
            //         |
            //         "
            // dst:    x
            auto x = dispatch_x();
            dst.write(x, src.read(idx.read(x)));
        });

        #define CREATE_REINEDX_SHADER(TYPE, SYMBOL) create_make_reindex_shader(device, FilterOp::TYPE, [](Var<T> a, Var<T> b){ return a SYMBOL b; })
        
        CREATE_REINEDX_SHADER(LESS, <);
        CREATE_REINEDX_SHADER(LESS_EQUAL, <=);
        CREATE_REINEDX_SHADER(GREATER, >);
        CREATE_REINEDX_SHADER(GREATER_EQUAL, >=);
        CREATE_REINEDX_SHADER(EQUAL, ==);
        CREATE_REINEDX_SHADER(NOT_EQUAL, !=);

        #undef CREATE_REINEDX_SHADER

    }

    static ShaderCollector *instance;
public:
    Shader1D<Buffer<uint>> reset_shader;
    Shader1D<Buffer<T>, Buffer<T>> copy_shader;
    Shader1D<Buffer<T>, Buffer<T>, BufferIndex> reindex_shader;
    Shader1D<Buffer<T>, Buffer<T>, BufferIndex> inverse_reindex_shader;
    unordered_map<FilterOp, Shader1D<BufferIndex, Buffer<uint>, Buffer<T>, T>> make_inverse_reindex_shader_map;

    static ShaderCollector *get_instance(luisa::compute::Device &device) {
        if (instance == nullptr) {
            instance = new ShaderCollector(device);
        }
        return instance;
    }
};

#define INSTANTIATE_SHADER(TYPE) template <> ShaderCollector<TYPE> *ShaderCollector<TYPE>::instance = nullptr;

INSTANTIATE_SHADER(int)
INSTANTIATE_SHADER(uint)
INSTANTIATE_SHADER(float)
INSTANTIATE_SHADER(long long)

#undef INSTANTIATE_SHADER