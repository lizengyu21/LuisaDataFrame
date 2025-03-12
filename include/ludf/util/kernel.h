#pragma once
#include <luisa/luisa-compute.h>
#include <ludf/core/type.h>
#include <ludf/core/type_dispatcher.h>
#include <ludf/core/hashmap.h>
#include <ludf/core/bitmap.h>

template <class T>
class ShaderCollector {
private:
    ShaderCollector() = delete;

    void create_make_reindex_shader(luisa::compute::Device &device, const FilterOp &op, auto &&filter) {
        using namespace luisa;
        using namespace luisa::compute;

        if (make_inverse_reindex_shader_map.find(op) != make_inverse_reindex_shader_map.end()) return;

        auto shader = device.compile<1>([&](BufferUInt indices, BufferUInt counter, BufferVar<T> data, Var<Bitmap> null_mask, Var<T> threshold){
            auto x = dispatch_x(); 
            auto pred = filter(data.read(x), threshold) & !null_mask->test(x);
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
        using namespace luisa;
        using namespace luisa::compute;
        
        reset_shader = device.compile<1>([](BufferVar<T> counter) {
            counter.write(dispatch_x(), cast<T>(0));
        });
        set_shader = device.compile<1>([](BufferVar<T> counter, Var<T> value) {
            counter.write(dispatch_x(), value);
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
        inverse_reindex_bitmap_shader = device.compile<1>([](Var<Bitmap> dst, Var<Bitmap> src, BufferVar<uint> idx){
            // dst[x] = src[idx[x]];
            // src: idx[x]
            //         |
            //         "
            // dst:    x
            auto x = dispatch_x();
            $if (src->test(idx.read(x))) {
                dst->set(x);
            } $else {
                dst->clear(x);
            };
        });
        arange_shader = device.compile<1>([](BufferVar<T> data){
            auto x = dispatch_x();
            data.write(x, cast<T>(x));
        });
        adjacent_diff_shader = device.compile<1>([](BufferVar<T> data, BufferVar<uint> result){
            auto x = dispatch_x();
            $if (data.read(x) != data.read(x + 1)) {
                result.write(x + 1, 1u);
            } $else {
                result.write(x + 1, 0u);
            };
        });
        aggregate_count_shader = device.compile<1>([](BufferVar<uint> result, BufferUInt indices){
            auto x = dispatch_x();
            result.atomic(indices.read(x)).fetch_add(1u);
        });
        adjacent_diff_index_shader = device.compile<1>([](BufferVar<uint> adjacent_diff_result, BufferVar<uint> indices, BufferVar<uint> result){
            auto x = dispatch_x();
            $if (adjacent_diff_result.read(x) == 1u) {
                result.write(indices.read(x) - 1, x);
            };
        });
        unique_count_shader = device.compile<1>([](BufferVar<uint> adjacent_diff_index_result, BufferVar<uint> result){
            auto x = dispatch_x();
            $if (x == 0u) {
                result.write(0u, adjacent_diff_index_result.read(0u));
            } $else {
                result.write(x, adjacent_diff_index_result.read(x) - adjacent_diff_index_result.read(x - 1));
            };
        });
        sum_to_mean_shader = device.compile<1>([](BufferVar<T> sum_data, BufferVar<uint> count_data, BufferVar<float> mean_data){
            auto x = dispatch_x();
            mean_data.write(x, cast<float>(sum_data.read(x)) / cast<float>(count_data.read(x)));
        });

        join_count_shader = device.compile<2>([](BufferVar<T> left, BufferVar<T> right, BufferVar<uint> count, Var<Bitmap> left_null_mask, Var<Bitmap> right_null_mask, Var<Bitmap> match_mask){
            auto xy = dispatch_id().xy();
            auto x = xy.x;
            auto y = xy.y;
            auto left_key = left.read(x);
            auto right_key = right.read(y);
            $if (left_null_mask->test(x) | right_null_mask->test(y)) {

            } $elif (left_key == right_key) { 
                count.atomic(x).fetch_add(1u); 
                $if (!match_mask->test(x)) {
                    match_mask->set(x);
                };
            };
        });
        join_reindex_shader = device.compile<2>([](BufferVar<T> left, BufferVar<T> right, Var<Bitmap> left_null_mask, Var<Bitmap> right_null_mask, Var<Bitmap> match_mask, BufferVar<uint> result_index_start, BufferVar<uint> slot_pointer, BufferVar<uint> result_left, BufferVar<uint> result_right){
            auto xy = dispatch_id().xy();
            auto x = xy.x;
            auto y = xy.y;
            auto left_key = left.read(x);
            auto right_key = right.read(y);
            $if (left_null_mask->test(x) | right_null_mask->test(y)) {

            } $elif (left_key == right_key) { 
                auto index_start = result_index_start.read(x);
                auto offset = slot_pointer.atomic(x).fetch_add(1u);
                result_left.write(index_start + offset, x);
                result_right.write(index_start + offset, y);
            };
        });
        join_match_mask_filter_shader = device.compile<1>([](Var<Bitmap> match_mask, Var<Bitmap> null_mask, BufferVar<uint> result_index_start, BufferUInt result_left, BufferUInt result_right){
            auto x = dispatch_x();
            auto idx = result_index_start.read(x);
            $if (!match_mask->test(x) & !null_mask->test(x)) {
                result_left.write(idx, x);
                result_right.write(idx, UINT_NULL);
            };
        });
        sum_shader = device.compile<1>([](BufferUInt count, Var<Bitmap> left_null_mask, BufferUInt result){
            auto x = dispatch_x();
            Shared<uint> block_sum{1};
            $if (thread_x() == 0u) { block_sum.write(0, 0); };
            sync_block();
            auto data = count.read(x);
            $if (data == 0u & !left_null_mask->test(x)) { 
                count.write(x, 1u);
                block_sum.atomic(0).fetch_add(1u); 
            } $elif (data != 0u) { block_sum.atomic(0).fetch_add(data); };
            sync_block();
            $if (thread_x() == 0u) { result.atomic(0).fetch_add(block_sum.read(0)); };
        });
        reindex_with_nullmask_shader = device.compile<1>([](BufferVar<T> dst, BufferVar<T> src, BufferVar<uint> indices, Var<Bitmap> origin_null_mask, Var<Bitmap> result_null_mask){
            // dst[idx[x]] = src[x];
            // src:   [x]
            //         |
            //         "
            // dst: idx[x]
            auto x = dispatch_x();
            auto idx = indices->read(x);
            $if (idx == UINT_NULL) {
                result_null_mask->set(x);
                dst.write(x, cast<T>(UINT_NULL));
            } $else {
                $if (origin_null_mask->test(idx)) {
                    dst.write(x, cast<T>(UINT_NULL));
                    result_null_mask->set(x);
                } $else {
                    dst.write(x, src.read(idx));
                };
            };
        });

        #define CREATE_REINEDX_SHADER(TYPE, SYMBOL) create_make_reindex_shader(device, FilterOp::TYPE, [](Var<T> a, Var<T> b){ return a SYMBOL b; })
        
        CREATE_REINEDX_SHADER(LESS, <);
        CREATE_REINEDX_SHADER(LESS_EQUAL, <=);
        CREATE_REINEDX_SHADER(GREATER, >);
        CREATE_REINEDX_SHADER(GREATER_EQUAL, >=);
        CREATE_REINEDX_SHADER(EQUAL, ==);
        CREATE_REINEDX_SHADER(NOT_EQUAL, !=);

        #undef CREATE_REINEDX_SHADER

        #define CREATE_AGG_SHADER(TYPE, type) \
            if (aggregate_shader_map.find(TYPE) == aggregate_shader_map.end()) \
                aggregate_shader_map[TYPE] = device.compile<1>([](BufferVar<T> data, BufferVar<T> result, BufferUInt indices, UInt size, Var<T> init_value){ \
                    Shared<T> block_sum{block_size_x()}; \
                    Shared<uint> block_start_index{1u}; \
                    Shared<uint> block_end_index{1u}; \
                    auto x = dispatch_x(); \
                    block_sum.write(thread_x(), init_value); \
                    $if (thread_x() == 0u) { block_start_index.write(0u, indices.read(x)); }; \
                    $if (thread_x() == block_size_x() - 1u | x == size - 1u) { block_end_index.write(0u, indices.read(x)); }; \
                    sync_block(); \
                    auto block_sum_id = indices.read(x) - block_start_index.read(0u); \
                    block_sum.atomic(block_sum_id).fetch_##type(data.read(x)); \
                    sync_block(); \
                    $if (thread_x() + block_start_index.read(0u) <= block_end_index.read(0u)) { \
                        result.atomic(thread_x() + block_start_index.read(0u)).fetch_##type(block_sum.read(thread_x())); \
                    }; \
                })


        CREATE_AGG_SHADER(AggeragateOp::SUM, add);
        CREATE_AGG_SHADER(AggeragateOp::MEAN, add);
        CREATE_AGG_SHADER(AggeragateOp::MAX, max);
        CREATE_AGG_SHADER(AggeragateOp::MIN, min);

        #undef CREATE_AGG_SHADER

    }

    static ShaderCollector *instance;
public:

    luisa::compute::Shader1D<luisa::compute::Buffer<T>> reset_shader;
    luisa::compute::Shader1D<luisa::compute::Buffer<T>, T> set_shader;
    luisa::compute::Shader1D<luisa::compute::Buffer<T>, luisa::compute::Buffer<T>> copy_shader;
    luisa::compute::Shader1D<luisa::compute::Buffer<T>, luisa::compute::Buffer<T>, BufferIndex> reindex_shader;
    luisa::compute::Shader1D<luisa::compute::Buffer<T>, luisa::compute::Buffer<T>, BufferIndex> inverse_reindex_shader;
    luisa::compute::Shader1D<Bitmap, Bitmap, BufferIndex> inverse_reindex_bitmap_shader;
    luisa::unordered_map<FilterOp, luisa::compute::Shader1D<BufferIndex, luisa::compute::Buffer<uint>, luisa::compute::Buffer<T>, Bitmap, T>> make_inverse_reindex_shader_map;
    luisa::unordered_map<AggeragateOp, luisa::compute::Shader1D<luisa::compute::Buffer<T>, luisa::compute::Buffer<T>, BufferIndex, uint, T>> aggregate_shader_map;
    luisa::compute::Shader1D<luisa::compute::Buffer<T>> arange_shader;
    luisa::compute::Shader1D<luisa::compute::Buffer<T>, BufferIndex> adjacent_diff_shader;
    luisa::compute::Shader1D<luisa::compute::Buffer<uint>, BufferIndex> aggregate_count_shader;
    luisa::compute::Shader1D<BufferIndex, BufferIndex, BufferIndex> adjacent_diff_index_shader;
    luisa::compute::Shader1D<BufferIndex, BufferIndex> unique_count_shader;
    luisa::compute::Shader1D<luisa::compute::Buffer<T>, BufferIndex, luisa::compute::Buffer<float>> sum_to_mean_shader;
    luisa::compute::Shader1D<luisa::compute::Buffer<T>, luisa::compute::Buffer<T>> apply_shader;

    luisa::compute::Shader2D<luisa::compute::Buffer<T>, luisa::compute::Buffer<T>, luisa::compute::Buffer<uint>, Bitmap, Bitmap, Bitmap> join_count_shader;
    luisa::compute::Shader2D<luisa::compute::Buffer<T>, luisa::compute::Buffer<T>, Bitmap, Bitmap, Bitmap, luisa::compute::Buffer<uint>, luisa::compute::Buffer<uint>, luisa::compute::Buffer<uint>, luisa::compute::Buffer<uint>> join_reindex_shader;
    luisa::compute::Shader1D<Bitmap, Bitmap, luisa::compute::Buffer<uint>, luisa::compute::Buffer<uint>, luisa::compute::Buffer<uint>> join_match_mask_filter_shader;

    luisa::compute::Shader1D<luisa::compute::Buffer<T>, luisa::compute::Buffer<T>, BufferIndex, Bitmap, Bitmap> reindex_with_nullmask_shader;
    luisa::compute::Shader1D<luisa::compute::Buffer<uint>, Bitmap, luisa::compute::Buffer<uint>> sum_shader;


    static ShaderCollector *get_instance(luisa::compute::Device &device) {
        if (instance == nullptr) {
            instance = new ShaderCollector(device);
        }
        return instance;
    }

    void create_apply_shader(luisa::compute::Device &device, void *apply_func_ptr) {
        using namespace luisa;
        using namespace luisa::compute;

        auto apply_func = *reinterpret_cast<Callable<T(T)>*>(apply_func_ptr);

        apply_shader = device.compile<1>([&](BufferVar<T> dst, BufferVar<T> src){
            auto x = dispatch_x();
            dst.write(x, apply_func(src.read(x)));
        });
    }

    template <class Ret, std::enable_if_t<!std::is_same_v<Ret, T>, int> = 0>
    luisa::compute::Shader1D<luisa::compute::Buffer<Ret>, luisa::compute::Buffer<T>> create_apply_Ret_T_shader(luisa::compute::Device &device, void *apply_func_ptr) {
        using namespace luisa;
        using namespace luisa::compute;

        auto apply_func = *reinterpret_cast<Callable<Ret(T)>*>(apply_func_ptr);

        auto apply_Ret_T_shader = device.compile<1>([&](BufferVar<Ret> dst, BufferVar<T> src){
            auto x = dispatch_x();
            dst.write(x, apply_func(src.read(x)));
        });

        return std::move(apply_Ret_T_shader);
    }
};

#define INSTANTIATE_SHADER(TYPE) template <> ShaderCollector<id_to_type<TYPE>> *ShaderCollector<id_to_type<TYPE>>::instance = nullptr;

INSTANTIATE_SHADER(TypeId::INT32)
INSTANTIATE_SHADER(TypeId::FLOAT32)
INSTANTIATE_SHADER(TypeId::UINT32)
// INSTANTIATE_SHADER(TypeId::INT64)

#undef INSTANTIATE_SHADER