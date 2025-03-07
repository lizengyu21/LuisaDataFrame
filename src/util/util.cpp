#include <ludf/util/util.h>
#include <luisa/backends/ext/cuda/lcub/device_scan.h>

BufferIndex inclusive_sum(luisa::compute::Device &device, luisa::compute::Stream &stream, BufferIndex &adjacent_diff_result) {
    using namespace luisa;
    using namespace luisa::compute;
    using namespace luisa::compute::cuda::lcub;
    
    size_t num_item = adjacent_diff_result.size();
    BufferIndex result = device.create_buffer<uint>(num_item);

    Buffer<int> temp_storage;
    size_t temp_storage_size = -1;

    DeviceScan::ExclusiveSum(temp_storage_size, adjacent_diff_result, result, num_item);
    temp_storage = device.create_buffer<int>(temp_storage_size);
    stream << DeviceScan::InclusiveSum(temp_storage, adjacent_diff_result, result, num_item);
    
    return std::move(result);
}