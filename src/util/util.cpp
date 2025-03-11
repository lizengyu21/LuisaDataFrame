#include <ludf/util/util.h>
#include <luisa/backends/ext/cuda/lcub/device_scan.h>
#include <ludf/core/type.h>
#include <ludf/core/hashmap.h>

BufferIndex inclusive_sum(luisa::compute::Device &device, luisa::compute::Stream &stream, BufferIndex &adjacent_diff_result) {
    using namespace luisa;
    using namespace luisa::compute;
    using namespace luisa::compute::cuda::lcub;
    
    size_t num_item = adjacent_diff_result.size();
    BufferIndex result = device.create_buffer<uint>(num_item);

    Buffer<int> temp_storage;
    size_t temp_storage_size = -1;

    DeviceScan::InclusiveSum(temp_storage_size, adjacent_diff_result, result, num_item);
    temp_storage = device.create_buffer<int>(temp_storage_size);
    stream << DeviceScan::InclusiveSum(temp_storage, adjacent_diff_result, result, num_item);
    
    return std::move(result);
}

BufferIndex exclusive_sum(luisa::compute::Device &device, luisa::compute::Stream &stream, BufferIndex &adjacent_diff_result) {
    using namespace luisa;
    using namespace luisa::compute;
    using namespace luisa::compute::cuda::lcub;
    
    size_t num_item = adjacent_diff_result.size();
    BufferIndex result = device.create_buffer<uint>(num_item);

    Buffer<int> temp_storage;
    size_t temp_storage_size = -1;

    DeviceScan::ExclusiveSum(temp_storage_size, adjacent_diff_result, result, num_item);
    temp_storage = device.create_buffer<int>(temp_storage_size);
    stream << DeviceScan::ExclusiveSum(temp_storage, adjacent_diff_result, result, num_item);
    
    return std::move(result);
}

BufferBase unique_count(luisa::compute::Device &device, luisa::compute::Stream &stream, BufferIndex &adjacent_diff_result, BufferIndex &indices, uint num_group) {
    using namespace luisa;
    using namespace luisa::compute;

    BufferIndex adjacent_diff_index_result = device.create_buffer<uint>(num_group);
    BufferBase result = device.create_buffer<BaseType>(num_group * sizeof(uint) / sizeof(BaseType));
    // std::cout << 1 << std::endl;
    // std::cout << adjacent_diff_result.size() << "  " << indices.size() << std::endl;
    stream << ShaderCollector<uint>::get_instance(device)->adjacent_diff_index_shader(adjacent_diff_result, indices, adjacent_diff_index_result).dispatch(indices.size());
    stream << synchronize();
    // std::cout << 2 << std::endl;
    uint total_item_count = indices.size();
    stream << adjacent_diff_index_result.view(adjacent_diff_index_result.size() - 1, 1).copy_from(&total_item_count) << synchronize();
    // std::cout << 3 << std::endl;
    stream << ShaderCollector<uint>::get_instance(device)->unique_count_shader(adjacent_diff_index_result, result.view().as<uint>()).dispatch(adjacent_diff_index_result.size());
    // std::cout << 4 << std::endl;
    return std::move(result);
}

