//
// Created by Mike on 2024/5/21.
//

#include <luisa/luisa-compute.h>
#include <luisa/core/clock.h>
// For the DSL sugar macros like $if.
// We exclude this header from <luisa-compute.h> to avoid pollution.
// So you have to include it explicitly to use the sugar macros.
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
// for std::cerr
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
// stb for image saving
#include <stb/stb_image_write.h>

#include <numeric>
#include <random>
// #include <luisa/backends/ext/cuda/lcub/device_scan.h>
// #include <luisa/backends/ext/cuda/lcub/device_radix_sort.h>

using namespace luisa;
using namespace luisa::compute;
// using namespace luisa::compute::cuda::lcub;

template<class T>
void print_buffer(Stream &stream, const Buffer<T> & buffer) {
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

template<class T>
class ShaderHandler {
public:
    bool has_compiled = false;
    Shader1D<Buffer<T>> arange_shader;
    Shader1D<uint, Buffer<T>, Buffer<T>> concat_shader;
    Shader1D<Buffer<T>, Buffer<int>> gen_mask_shader;
    Shader1D<Buffer<uint>> reset_shader;
    Shader1D<Buffer<T>, Buffer<uint>, Buffer<T>, Buffer<int>> filter_by_mask_shader;
    Shader1D<Buffer<T>, Buffer<T>> copy_shader;
    Shader1D<Buffer<uint>, Buffer<uint>> uint_copy_shader;
    Shader1D<Buffer<T>, Buffer<T>, Buffer<uint>> reindex_shader;
    Shader1D<Buffer<T>, Buffer<T>, Buffer<uint>> inverse_reindex_shader;

    #define DEFINE_MASK_SHADER(type) \
        Shader1D<Buffer<T>, Buffer<int>, T> mask_shader_##type

    DEFINE_MASK_SHADER(LESS);
    DEFINE_MASK_SHADER(LESS_EQUAL);
    DEFINE_MASK_SHADER(GREATER);
    DEFINE_MASK_SHADER(GREATER_EQUAL);
    DEFINE_MASK_SHADER(EQUAL);
    DEFINE_MASK_SHADER(NOT_EQUAL);

    #define DEFINE_GEN_REINDEX_SHADER(type) \
        Shader1D<Buffer<uint>, Buffer<uint>, Buffer<T>, T> gen_reindex_shader_##type

    DEFINE_GEN_REINDEX_SHADER(LESS);
    DEFINE_GEN_REINDEX_SHADER(LESS_EQUAL);
    DEFINE_GEN_REINDEX_SHADER(GREATER);
    DEFINE_GEN_REINDEX_SHADER(GREATER_EQUAL);
    DEFINE_GEN_REINDEX_SHADER(EQUAL);
    DEFINE_GEN_REINDEX_SHADER(NOT_EQUAL);

    void compile_shader(Device &device) {
        if (has_compiled) return;

        Kernel1D arange_kernel = [](BufferVar<T> result){
            auto id = dispatch_x();
            result.write(id, Var<T>(id));
        };

        Kernel1D concat_kernel = [](UInt start_id, BufferVar<T> other, BufferVar<T> result) {
            auto id = dispatch_x() + start_id;
            result.write(id, other.read(dispatch_x()));
        };
        #define GEN_MASK_KERNEL(type, symbol) \
            Kernel1D mask_kernel_##type = [] (BufferVar<T> data, BufferInt mask, Var<T> threshold) { \
                auto idx = dispatch_x(); \
                $if (data.read(idx) symbol threshold) { \
                    mask.write(idx, def(1)); \
                } $else { \
                    mask.write(idx, def(0)); \
                }; \
            }; \
            mask_shader_##type = device.compile(mask_kernel_##type)

        GEN_MASK_KERNEL(LESS, <);
        GEN_MASK_KERNEL(LESS_EQUAL, <=);
        GEN_MASK_KERNEL(GREATER, >);
        GEN_MASK_KERNEL(GREATER_EQUAL, >=);
        GEN_MASK_KERNEL(EQUAL, ==);
        GEN_MASK_KERNEL(NOT_EQUAL, !=);

        Kernel1D reset_kernel = [](BufferVar<uint> counter) {
            counter.write(0, 0u);
        };

        Kernel1D filter_by_mask_kernel = [](BufferVar<T> buffer, BufferUInt counter, BufferVar<T> data, BufferInt mask) {
            auto x = dispatch_x();
            auto pred = mask.read(x) == 1;

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
                buffer->write(global_index, data.read(x));
            };
        };

        Kernel1D copy_kernel = [](BufferVar<T> dst, BufferVar<T> src) {
            auto x = dispatch_x();
            dst.write(x, src.read(x));
        };

        Kernel1D uint_copy_kernel = [](BufferUInt dst, BufferUInt src) {
            auto x = dispatch_x();
            dst.write(x, src.read(x));
        };

        Kernel1D reindex_kernel = [](BufferVar<T> dst, BufferVar<T> src, BufferUInt indices) {
            auto x = dispatch_x();
            dst.write(x, src.read(indices.read(x)));
        };

        Kernel1D inverse_reindex_kernel = [](BufferVar<T> dst, BufferVar<T> src, BufferUInt indices) {
            auto x = dispatch_x();
            dst.write(indices.read(x), src.read(x));
        };

        #define GEN_GEN_REINDEX_KERNEL(type, symbol) \
            Kernel1D gen_reindex_kernel_##type = [](BufferUInt indices, BufferUInt counter, BufferVar<T> data, Var<T> threshold) { \
                auto x = dispatch_x(); \
                auto pred = data.read(x) symbol threshold; \
                Shared<uint> index{1}; \
                $if (thread_x() == 0u) { index.write(0u, 0u); }; \
                sync_block(); \
                auto local_index = def(0u); \
                $if (pred) { local_index = index.atomic(0).fetch_add(1u); }; \
                sync_block(); \
                $if (thread_x() == 0u) { \
                    auto local_count = index.read(0u); \
                    auto global_offset = counter->atomic(0u).fetch_add(local_count); \
                    index.write(0u, global_offset); \
                }; \
                sync_block(); \
                $if (pred) { \
                    auto global_index = index.read(0u) + local_index; \
                    indices->write(global_index, x); \
                }; \
            }; \
            gen_reindex_shader_##type = device.compile(gen_reindex_kernel_##type)

        GEN_GEN_REINDEX_KERNEL(LESS, <);
        GEN_GEN_REINDEX_KERNEL(LESS_EQUAL, <=);
        GEN_GEN_REINDEX_KERNEL(GREATER, >);
        GEN_GEN_REINDEX_KERNEL(GREATER_EQUAL, >=);
        GEN_GEN_REINDEX_KERNEL(EQUAL, ==);
        GEN_GEN_REINDEX_KERNEL(NOT_EQUAL, !=);

        arange_shader = device.compile(arange_kernel);
        concat_shader = device.compile(concat_kernel);
        reset_shader = device.compile(reset_kernel);
        filter_by_mask_shader = device.compile(filter_by_mask_kernel);
        copy_shader = device.compile(copy_kernel);
        uint_copy_shader = device.compile(uint_copy_kernel);
        reindex_shader = device.compile(reindex_kernel);
        inverse_reindex_shader = device.compile(inverse_reindex_kernel);
        has_compiled = true;
    }
};

enum FilterOperation {LESS=0, LESS_EQUAL, GREATER, GREATER_EQUAL, EQUAL, NOT_EQUAL};
enum AggeragateOperation {SUM=0, MEAN, MAX, MIN, COUNT};

template<class T>
class Tensor {

public:
    Buffer<T> _data;
    static ShaderHandler<T> _shader_handler;
    // Tensor() = delete;
    Tensor() = default;
    Tensor(Buffer<T> &&data) : _data(std::move(data)) {}
    Tensor(Device &device, size_t size) : _data(device.create_buffer<T>(size)) {}
    Tensor(Device &device, Stream &stream, luisa::vector<T> data) : Tensor(device, data.size()) {
        stream << _data.copy_from(data.data()) << synchronize();
    }
    Tensor(Device &device, Stream &stream, const void *data, size_t size) : Tensor(device, size) {
        stream << _data.copy_from(data) << synchronize();
    }

    size_t size() const {
        return _data.size();
    }

    void concat(Device &device, Stream &stream, const Tensor<T> &other) {
        if (size() == 0) {
            _data = device.create_buffer<T>(other.size());
            stream << _data.copy_from(other._data);
            return;
        }

        auto result = device.create_buffer<T>(size() + other.size());

        stream << _shader_handler.concat_shader(0u, _data, result).dispatch(size())
                << _shader_handler.concat_shader(size(), other._data, result).dispatch(other.size());

        _data = std::move(result);
    }

    Buffer<int> gen_mask(Device &device, Stream &stream, const FilterOperation &operation, T threshold) {
        Buffer<int> mask = device.create_buffer<int>(size());
        switch (operation) {
            #define OP2SHADER(type) \
                case FilterOperation::type: { \
                    stream << _shader_handler.mask_shader_##type(_data, mask, threshold).dispatch(size()); \
                    break; \
                }
            
            OP2SHADER(LESS)
            OP2SHADER(LESS_EQUAL)
            OP2SHADER(GREATER)
            OP2SHADER(GREATER_EQUAL)
            OP2SHADER(EQUAL)
            OP2SHADER(NOT_EQUAL)
            
            #undef OP2SHADER
            default:
                break;
        }
        return std::move(mask);
    }

    void filter_by_mask(Device &device, Stream &stream, const Buffer<int> &mask) {
        Buffer<T> buffer = device.create_buffer<T>(size());
        Buffer<uint> counter = device.create_buffer<uint>(1u);
        stream << _shader_handler.reset_shader(counter).dispatch(1)
                << _shader_handler.filter_by_mask_shader(buffer, counter, _data, mask).dispatch(size());
        uint count;
        stream << counter.copy_to(&count) << synchronize();
        if (count == 0) {
            _data = Buffer<T>();
            return;
        }
        _data = device.create_buffer<T>(count);
        stream << _shader_handler.copy_shader(_data, buffer).dispatch(count);
    }

    Buffer<uint> gen_reindex(Device &device, Stream &stream, const FilterOperation &operation, T threshold) {
        Buffer<uint> indices = device.create_buffer<uint>(size());
        Buffer<uint> counter = device.create_buffer<uint>(1u);
        stream << _shader_handler.reset_shader(counter).dispatch(1);
        // (BufferUInt indices, BufferUInt counter, BufferVar<T> data, Var<T> threshold)
        switch (operation) {
            #define OP2SHADER(type) \
                case FilterOperation::type: { \
                    stream << _shader_handler.gen_reindex_shader_##type(indices, counter, _data, threshold).dispatch(size()); \
                    break; \
                }
            
            OP2SHADER(LESS)
            OP2SHADER(LESS_EQUAL)
            OP2SHADER(GREATER)
            OP2SHADER(GREATER_EQUAL)
            OP2SHADER(EQUAL)
            OP2SHADER(NOT_EQUAL)

            #undef OP2SHADER
            default:
                break;
        }
        uint count;
        stream << counter.copy_to(&count) << synchronize();
        if (count == 0) {
            return Buffer<uint>();
        }
        auto res_indices = device.create_buffer<uint>(count);
        stream << _shader_handler.uint_copy_shader(res_indices, indices).dispatch(count);
        return std::move(res_indices);
    }

    void filter_by_reindex(Device &device, Stream &stream, const Buffer<uint> &indices) {
        auto count = indices.size();
        if (count == 0) {
            _data = Buffer<T>();
            return;
        }
        auto res_data = device.create_buffer<T>(count);
        stream << _shader_handler.reindex_shader(res_data, _data, indices).dispatch(count);
        _data = std::move(res_data);
    }

    void filter_by_inverse_reindex(Device &device, Stream &stream, const Buffer<uint> &indices) {
        auto count = indices.size();
        if (count == 0) {
            _data = Buffer<T>();
            return;
        }
        auto res_data = device.create_buffer<T>(count);
        stream << _shader_handler.inverse_reindex_shader(res_data, _data, indices).dispatch(count);
        _data = std::move(res_data);
    }

    static void compile_shader(Device &device) {
        _shader_handler.compile_shader(device);
    }

    static Tensor<T> arange(Device &device, Stream &stream, size_t size) {
        auto result = device.create_buffer<T>(size);
        stream  << _shader_handler.arange_shader(result).dispatch(result.size());
        return std::move(result);
    }
};

template<class T>
ShaderHandler<T> Tensor<T>::_shader_handler;

enum TensorDataType {INT=0, FLOAT=1};


class Table {
public:
    void _create_column(const luisa::string &name, const TensorDataType &type) {
        _column_type.insert({name, type});

        switch (type) {
            case TensorDataType::INT:
                _column_int.insert({name, Tensor<int>()});
                break;
            case TensorDataType::FLOAT:
                _column_float.insert({name, Tensor<float>()});
                break;
            default:
                break;
        }
    }

    luisa::unordered_map<luisa::string, Tensor<int>> _column_int;
    luisa::unordered_map<luisa::string, Tensor<float>> _column_float;
    
    luisa::unordered_map<luisa::string, TensorDataType> _column_type;

    Device &_device;
    Stream &_stream;

public:


    Table() = delete;

    Table(Device &device, Stream &stream) : _device(device), _stream(stream) {
        Tensor<int>::compile_shader(device);
        Tensor<float>::compile_shader(device);
    }

    Table start_query() {
        auto rows_count = size();
        Table query_table(_device, _stream);
        query_table._column_type = _column_type;

        if (rows_count == 0) return std::move(query_table);

        for (auto it = _column_int.cbegin(); it != _column_int.cend(); ++it) {
            query_table._column_int[it->first] = _device.create_buffer<int>(rows_count);
            _stream << query_table._column_int[it->first]._data.copy_from(it->second._data);
        }

        for (auto it = _column_float.cbegin(); it != _column_float.cend(); ++it) {
            query_table._column_float[it->first] = _device.create_buffer<float>(rows_count);
            _stream << query_table._column_float[it->first]._data.copy_from(it->second._data);
        }

        return std::move(query_table);
    }

    void insert_column(const luisa::string &name, const luisa::variant<luisa::vector<int>, luisa::vector<float>> &data) {
        const auto &type = _column_type[name];
        switch (type) {
            case TensorDataType::INT: {
                auto &col = _column_int.find(name)->second;
                auto data_int = luisa::get<luisa::vector<int>>(data);
                col.concat(_device, _stream, Tensor<int>(_device, _stream, data_int));
                break;
            }
            case TensorDataType::FLOAT: {
                auto &col = _column_float.find(name)->second;
                auto data_float = luisa::get<luisa::vector<float>>(data);
                col.concat(_device, _stream, Tensor<float>(_device, _stream, data_float));
                break;
            }
            default:
                break;
        }
    }

    void create_table(luisa::vector<luisa::string> col_names, luisa::vector<TensorDataType> col_types) {
        assert(col_names.size() == col_types.size());
        for (auto i = 0; i < col_names.size(); ++i) {
            _create_column(col_names[i], col_types[i]);
        }
    }

    size_t size() const noexcept {
        if (_column_int.begin() != _column_int.end()) {
            return _column_int.begin()->second.size();
        }
        if (_column_float.begin() != _column_float.end()) {
            return _column_float.begin()->second.size();
        }
        return 0;
    }

    // void where(const luisa::string &name, const FilterOperation &operation, luisa::variant<int, float> threshold) {
    //     const auto &type = _column_type[name];
    //     Buffer<int> mask;
    //     uint count;
    //     switch (type) {
    //         case TensorDataType::INT: {
    //             auto &col = _column_int.find(name)->second;
    //             mask = col.gen_mask(_device, _stream, operation, luisa::get<int>(threshold));
    //             col.filter_by_mask(_device, _stream, mask);
    //             break;
    //         }

    //         case TensorDataType::FLOAT: {
    //             auto &col = _column_float.find(name)->second;
    //             mask = col.gen_mask(_device, _stream, operation, luisa::get<float>(threshold));
    //             col.filter_by_mask(_device, _stream, mask);
    //             break;
    //         }

    //         default:
    //             unreachable("");
    //             break;
    //     }

    //     for (auto it = _column_int.begin(); it != _column_int.end(); ++it) {
    //         if (it->first == name) continue;
    //         it->second.filter_by_mask(_device, _stream, mask);
    //     }

    //     for (auto it = _column_float.begin(); it != _column_float.end(); ++it) {
    //         if (it->first == name) continue;
    //         it->second.filter_by_mask(_device, _stream, mask);
    //     }

    //     _stream << synchronize();

    // }

    void where(const luisa::string &name, const FilterOperation &operation, luisa::variant<int, float> threshold) {
        assert(_column_type.find(name) != _column_type.end());
        const auto &type = _column_type[name];
        Buffer<uint> indices;
        switch (type) {
            case TensorDataType::INT: {
                auto &col = _column_int.find(name)->second;
                indices = col.gen_reindex(_device, _stream, operation, luisa::get<int>(threshold));
                break;
            }

            case TensorDataType::FLOAT: {
                auto &col = _column_float.find(name)->second;
                indices = col.gen_reindex(_device, _stream, operation, luisa::get<float>(threshold));
                break;
            }

            default:
                unreachable("");
                break;
        }
        for (auto it = _column_int.begin(); it != _column_int.end(); ++it) {
            it->second.filter_by_reindex(_device, _stream, indices);
        }

        for (auto it = _column_float.begin(); it != _column_float.end(); ++it) {
            it->second.filter_by_reindex(_device, _stream, indices);
        }

        // _stream << synchronize();
    }

    void _group_by(const luisa::string &name, const AggeragateOperation &operation) {
        assert(_column_type.find(name) != _column_type.end());
        const auto &type = _column_type[name];
        switch (type) {
        case TensorDataType::INT: {
            break;
        }
        
        default:
            LUISA_ERROR("not implemented!");
            break;
        }
    }

    void print_table() const noexcept {
        std::cout << "Table size: " << size() << " rows\n";
        for (auto it = _column_int.cbegin(); it != _column_int.cend(); ++it) {
            std::cout << it->first << ": ";
            print_buffer<int>(_stream, it->second._data);
        }

        for (auto it = _column_float.cbegin(); it != _column_float.cend(); ++it) {
            std::cout << it->first << ": ";
            print_buffer<float>(_stream, it->second._data);
        }
    }

};

void load_from_TRD_Dalyr(Table &table) {

    luisa::vector<luisa::string> col_names = {"stkcd", "opnprc", "loprc", "clsprc", "hiprc"};
    luisa::vector<TensorDataType> col_types = {TensorDataType::INT, TensorDataType::FLOAT, TensorDataType::FLOAT, TensorDataType::FLOAT, TensorDataType::FLOAT};

    table.create_table(col_names, col_types);

    for (int i = 0; i < 6; ++i) {
        std::string filename = "./data/TRD_Dalyr" + std::to_string(i) + ".csv";
        std::cout << "loading file " << filename << std::endl;
        std::fstream file(filename);
        luisa::vector<int> stkcd;
        luisa::vector<uint> trddt_timestamp;
        luisa::vector<float> opnprc, hiprc, loprc, clsprc;

        if (!file.is_open()) {
            std::cerr << "Failed to open file." << std::endl;
            return;
        }
        luisa::string line;
        getline(file, line);

        auto remove_quotes = [](const luisa::string &s) {
            return s.substr(1, s.size() - 2).c_str();
        };

        int total = 10;
        int stride = 1000;

        int cnt = -1;
        while (getline(file, line)) {
            ++cnt;
            // if (cnt > stride * total) break;
            // if (cnt % stride != 0) continue;
            std::stringstream ss(line);
            luisa::string stkcd_value, trddt_value, opnprc_value, hiprc_value, loprc_value, clsprc_value;

            getline(ss, stkcd_value, ',');
            getline(ss, trddt_value, ',');
            getline(ss, opnprc_value, ',');
            getline(ss, hiprc_value, ',');
            getline(ss, loprc_value, ',');
            getline(ss, clsprc_value, ',');

            stkcd.push_back(std::stoi(remove_quotes(stkcd_value)));
            opnprc.push_back(std::stof(remove_quotes(opnprc_value))); // 将 Opnprc 转换为 float
            loprc.push_back(std::stof(remove_quotes(loprc_value)));   // 将 Loprc 转换为 float
            clsprc.push_back(std::stof(remove_quotes(clsprc_value)));
            hiprc.push_back(std::stof(remove_quotes(hiprc_value)));
        }


        table.insert_column("stkcd", stkcd);
        table.insert_column("opnprc", opnprc);
        table.insert_column("loprc", loprc);
        table.insert_column("clsprc", clsprc);
        table.insert_column("hiprc", hiprc);
    }
}

template<class T>
void print_vector(const luisa::vector<T> &data) {
    std::cout << "SIZE[" << data.size() << "]: ";
    for (const auto &it : data) {
        std::cout << it << ", ";
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <backend>\n";
        return 1;
    }

    // Step 1.1: Create a context
    Context context{argv[0]};

    // Step 1.2: Load the CUDA backend plug-in and create a device
    Device device = context.create_device(argv[1]);

    // Step 2.1: Create a stream for command submission
    Stream stream = device.create_stream();
    Clock clock;
    Table table(device, stream);
    

    clock.tic();
    load_from_TRD_Dalyr(table);
    LUISA_INFO("Load from CSV file in {} ms.", clock.toc());

    table.print_table();
    
    // clock.tic();
    // table._group_by("stkcd", AggeragateOperation::COUNT);
    // LUISA_INFO("group by in {} ms.", clock.toc());

    // table.print_table();
    int count = std::stoi(argv[2]);
    double elapsed_time = 0.0;
    std::cout << "Run test " << count << " times.\n";
    for (int i = 0; i < count; ++i) {
        clock.tic();
        auto query_table = table.start_query();
        query_table.where("stkcd", FilterOperation::LESS, i);
        elapsed_time += clock.toc();
        // std::cout << "stkcd LESS " << i << " total rows: " << query_table.size() << std::endl;
    }
    // table.where("stkcd", FilterOperation::LESS, 10);

    LUISA_INFO("WHERE {} times in {} ms.", count, elapsed_time);
    
    // exit(0);

    std::cout << "End" << std::endl;
}
