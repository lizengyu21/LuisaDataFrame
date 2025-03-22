#pragma once
#include <luisa/luisa-compute.h>
#include <ludf/column/column.h>
#include <ludf/core/type.h>
#include <ludf/util/util.h>
#include <any>
#include <typeinfo>
#include <ludf/util/printer.h>

class Table {
    luisa::compute::Device &_device;
    luisa::compute::Stream &_stream;
    Printer printer;

    void create_column(const luisa::string &name, Column &&col) {
        if (_columns.find(name) != _columns.end()) return;
        _columns.insert({name, std::move(col)});
    }

public:
    luisa::unordered_map<luisa::string, Column> _columns;

    Table(luisa::compute::Device &device, luisa::compute::Stream &stream) : _device(device), _stream(stream) {}

    void create_column(const luisa::string &name, DataType dtype) {
        if (_columns.find(name) != _columns.end()) return;
        _columns.insert({name, Column{dtype}});
    }

    Table query() {
        Table t{_device, _stream};
        for (auto it = _columns.cbegin(); it != _columns.cend(); ++it) {
            t.create_column(it->first, it->second.clone(_device, _stream));
        }
        return std::move(t);
    }

    void precompile() {
        ShaderCollector<id_to_type<TypeId::FLOAT32>>::get_instance(_device);
        ShaderCollector<id_to_type<TypeId::INT32>>::get_instance(_device);
        ShaderCollector<id_to_type<TypeId::UINT32>>::get_instance(_device);
        
    }

    void append_column(const luisa::string &name, void *data, size_t size_byte) {
        if (_columns.find(name) == _columns.end()) {
            LUISA_WARNING("APPEND COLUMN SKIP: column not found. name: {}", name);
            return;
        }
        if (size_byte == 0) return;
        Column &col = _columns[name];
        const auto &type = col.dtype();
        Column data_col{type};
        data_col.load(_device, _stream, data, size_byte, true);
        type_dispatcher(type, concat_column{}, _device, _stream, col, data_col);
    }

    template <class T>
    void append_column(const luisa::string &name, luisa::vector<T> data) {
        append_column(name, data.data(), data.size_bytes());
    }

    Table *interval(luisa::string group_col_name, luisa::string ts_col_name = "", uint span = 60 * 60 * 24 * 2,  const luisa::vector<AggeragateOp> &agg_op_vec = {}) {
        using namespace luisa;
        using namespace luisa::compute;
        unordered_map<string, vector<AggeragateOp>> agg_op_map;

        for (auto it = _columns.begin(); it != _columns.end(); ++it) {
            if (it->first == group_col_name) continue;
            agg_op_map[it->first] = agg_op_vec;
        }

        return interval(group_col_name, ts_col_name, span, agg_op_map);
    }

    /*
    * @brief: interval function
    * @param: name: column name
    * @param: span: interval span in seconds
    * @return: Table pointer
    */
    Table *interval(luisa::string group_col_name, luisa::string ts_col_name = "", uint span = 60 * 60 * 24 * 2, const luisa::unordered_map<luisa::string, luisa::vector<AggeragateOp>> &agg_op_map = {}) {
        using namespace luisa;
        using namespace luisa::compute;

        if (_columns.find(group_col_name) == _columns.end()) {
            LUISA_WARNING("INTERVAL SKIP: column not found. group_col_name: {}", group_col_name);
            return this;
        }
        if (_columns[group_col_name].size() == 0) return this;

        if (group_col_name == ts_col_name) {
            LUISA_WARNING("INTERVAL SKIP: group_col_name == ts_col_name");
            return this;
        }

        if (ts_col_name == "") {
            for (auto it = _columns.begin(); it != _columns.end(); ++it) {
                if (it->first == group_col_name) continue;
                Column &col = it->second;
                const auto &type = col.dtype();
                if (type == TypeId::TIMESTAMP) {
                    ts_col_name = it->first;
                    break;
                }
            }
        } else {
            if (_columns.find(ts_col_name) == _columns.end()) {
                LUISA_WARNING("INTERVAL SKIP: column not found. ts_col_name: {}", ts_col_name);
                return this;
            }
            if (_columns[ts_col_name].dtype().id() != TypeId::TIMESTAMP) {
                LUISA_WARNING("INTERVAL SKIP: column type not match. name: {}, desired type TIMESTAMP", ts_col_name);
                return this;
            }
        }

        if (ts_col_name == "") {
            LUISA_WARNING("INTERVAL SKIP: no timestamp column found.");
            return this;
        }

        // LUISA_INFO("Interval: group_col_name: {}, ts_col_name: {}, span: {}", group_col_name, ts_col_name, span);

        auto &partition_col = _columns[group_col_name];
        auto &ts_col = _columns[ts_col_name];

        if (partition_col.dtype().id() != TypeId::INT32 && partition_col.dtype().id() != TypeId::UINT32) {
            LUISA_WARNING("INTERVAL SKIP: column type not match. name: {}, desired type INT32 or UINT32", group_col_name);
            return this;
        }

        if (partition_col.size() == 0) {
            LUISA_WARNING("INTERVAL SKIP: column size is 0. name: {}", group_col_name);
            return this;
        }

        // 采用和3DGS排序相同的策略，将ID放到高位，时间戳放到低位，拼成一个64位无符号整数
        auto merged = merge_id_ts(_device, _stream, partition_col, ts_col);

        auto indices = sort_u64_buffer(_device, _stream, merged);

        BufferBase sorted_id = _device.create_buffer<BaseType>(merged.size());
        _stream << ShaderCollector<BaseType>::get_instance(_device)->get_high_32_shader(merged, sorted_id).dispatch(merged.size());

        auto id_adjacent_diff_result = adjacent_diff_buffer<BaseType>(_device, _stream, sorted_id);

        auto id_inclusive_sum_result = inclusive_sum(_device, _stream, id_adjacent_diff_result);

        uint num_group;
        _stream << id_inclusive_sum_result.view(id_inclusive_sum_result.size() - 1, 1).copy_to(&num_group) << synchronize();
        ++num_group;

        Buffer<uint> start_ts = _device.create_buffer<uint>(num_group);

        _stream << ShaderCollector<uint>::get_instance(_device)->get_start_time_from_64_lo_shader(merged, id_adjacent_diff_result, id_inclusive_sum_result, start_ts).dispatch(merged.size());

        BufferBase total_start_ts = _device.create_buffer<BaseType>(merged.size() * sizeof(uint) / sizeof(BaseType));
        _stream << ShaderCollector<uint>::get_instance(_device)->inverse_reindex_shader(total_start_ts.view().as<uint>(), start_ts, id_inclusive_sum_result).dispatch(merged.size());

        _stream << ShaderCollector<uint>::get_instance(_device)->compute_time_block_id_from_64_lo_shader(merged, total_start_ts.view().as<uint>(), span).dispatch(merged.size());


        auto total_adjacent_diff_result = adjacent_diff_buffer<uint64>(_device, _stream, merged);

        auto total_inclusive_sum_result = inclusive_sum(_device, _stream, total_adjacent_diff_result);

        BufferBase total_end_ts = _device.create_buffer<BaseType>(merged.size() * sizeof(uint) / sizeof(BaseType));
        _stream << ShaderCollector<uint>::get_instance(_device)->compute_start_and_end_ts_from_64_lo_shader(merged, total_start_ts.view().as<uint>(), total_end_ts.view().as<uint>(), span).dispatch(merged.size());

        uint total_size;
        _stream << total_inclusive_sum_result.view(total_inclusive_sum_result.size() - 1, 1).copy_to(&total_size) << synchronize();
        ++total_size;
   
        partition_col.load(std::move(sorted_id));
        type_dispatcher(partition_col.dtype(), reindex{}, _device, _stream, partition_col, total_inclusive_sum_result, total_size);
        luisa::unordered_map<luisa::string, Column> res_columns;

        res_columns.insert({group_col_name, std::move(partition_col)});

        Column start_ts_col{TypeId::TIMESTAMP};
        start_ts_col.load(std::move(total_start_ts));
        type_dispatcher(start_ts_col.dtype(), reindex{}, _device, _stream, start_ts_col, total_inclusive_sum_result, total_size);
        res_columns.insert({"_start_ts", std::move(start_ts_col)});

        Column end_ts_col{TypeId::TIMESTAMP};
        end_ts_col.load(std::move(total_end_ts));
        type_dispatcher(end_ts_col.dtype(), reindex{}, _device, _stream, end_ts_col, total_inclusive_sum_result, total_size);
        res_columns.insert({"_end_ts", std::move(end_ts_col)});

        BufferBase each_group_count;

        for (auto it = agg_op_map.begin(); it != agg_op_map.end(); ++it) {
            if (it->first == group_col_name || it->first == ts_col_name) [[unlikely]] continue;
            if (_columns.find(it->first) == _columns.end()) [[unlikely]] continue;

            auto cur_kv = _columns.find(it->first);
            const string &current_col_name = cur_kv->first;
            Column &current_col = cur_kv->second;
            const auto &current_col_type = current_col.dtype();

            type_dispatcher(current_col_type, inverse_reindex{}, _device, _stream, current_col, indices);
            for (const auto &agg_op : it->second) {
                string new_col_name = agg_op_string(agg_op) + "(" + current_col_name + ")";

                auto res_col = type_dispatcher(current_col_type, aggregate_column{}, _device, _stream, current_col, agg_op, total_inclusive_sum_result, total_size);
                res_col._null_mask.init_zero(_device, _stream, total_size, ShaderCollector<uint>::get_instance(_device)->set_shader);

                _stream << ShaderCollector<uint>::get_instance(_device)->reindex_bitmap_with_null_shader(res_col._null_mask, current_col._null_mask, total_inclusive_sum_result).dispatch(current_col.size());

                if (each_group_count.size() == 0 && (agg_op == AggeragateOp::MEAN || agg_op == AggeragateOp::COUNT)) {
                    each_group_count = unique_count(_device, _stream, total_adjacent_diff_result, total_inclusive_sum_result, total_size);
                }

                if (agg_op == AggeragateOp::MEAN) {
                    type_dispatcher(current_col_type, sum_to_mean{}, _device, _stream, res_col, each_group_count.view().as<uint>());
                }

                if (agg_op == AggeragateOp::COUNT) {
                    res_col.load(_device, _stream, each_group_count);
                }

                res_columns.insert({new_col_name, std::move(res_col)});
            }
            _columns.erase(cur_kv);
        }

        _columns = std::move(res_columns);
        return this;
    }

    Table *where(const luisa::string &name, const FilterOp op, std::any threshold) {
        if (_columns.find(name) == _columns.end()) return this;
        Column &col = _columns[name];
        const auto &type = col.dtype();
        auto reindex = type_dispatcher(type, make_inverse_reindex{}, _device, _stream, col, op, threshold);
        for (auto it = _columns.begin(); it != _columns.end(); ++it) {
            type_dispatcher(it->second.dtype(), inverse_reindex{}, _device, _stream, it->second, reindex);
        }
        return this;
    }

    Table *group_by(const luisa::string &name, const luisa::vector<AggeragateOp> &agg_op_vec = {}) {
        using namespace luisa;
        using namespace luisa::compute;
        unordered_map<string, vector<AggeragateOp>> agg_op_map;

        for (auto it = _columns.begin(); it != _columns.end(); ++it) {
            if (it->first == name) continue;
            agg_op_map[it->first] = agg_op_vec;
        }
        return group_by(name, agg_op_map);
    }

    Table *group_by(const luisa::string &name, const luisa::unordered_map<luisa::string, luisa::vector<AggeragateOp>> &agg_op_map = {}) {
        using namespace luisa;
        using namespace luisa::compute;

        if (_columns.find(name) == _columns.end()) [[unlikely]] return this;

        Column &col = _columns[name];
        if (col.size() == 0) return this;

        const auto &type = col.dtype();
        Column sorted_result{type};
        auto indices = type_dispatcher(type, sort_column{}, _device, _stream, col, sorted_result);
        auto adjacent_diff_result = type_dispatcher(type, adjacent_diff{}, _device, _stream, sorted_result);
        auto inclusive_sum_result = inclusive_sum(_device, _stream, adjacent_diff_result);
        
        uint num_group;
        _stream << inclusive_sum_result.view(inclusive_sum_result.size() - 1, 1).copy_to(&num_group) << synchronize();
        ++num_group;

        if (sorted_result._null_mask._data.size() != 0)
            _stream << ShaderCollector<uint>::get_instance(_device)->filter_set_shader(inclusive_sum_result, sorted_result._null_mask, UINT_NULL).dispatch(inclusive_sum_result.size());


        // type_dispatcher(type, inverse_reindex{}, _device, _stream, col, indices);
        col = std::move(sorted_result);
        type_dispatcher(type, reindex{}, _device, _stream, col, inclusive_sum_result, num_group);
        luisa::unordered_map<luisa::string, Column> res_columns;

        res_columns.insert({name, std::move(col)});

        BufferBase each_group_count;

        for (auto it = agg_op_map.begin(); it != agg_op_map.end(); ++it) {
            if (it->first == name) [[unlikely]] continue;
            if (_columns.find(it->first) == _columns.end()) [[unlikely]] continue;

            auto cur_kv = _columns.find(it->first);
            const string &current_col_name = cur_kv->first;
            Column &current_col = cur_kv->second;
            const auto &current_col_type = current_col.dtype();

            type_dispatcher(current_col_type, inverse_reindex{}, _device, _stream, current_col, indices);
            for (const auto &agg_op : it->second) {
                string new_col_name = agg_op_string(agg_op) + "(" + current_col_name + ")";
                // print_buffer(_stream, current_col._null_mask._data.view());
                auto res_col = type_dispatcher(current_col_type, aggregate_column{}, _device, _stream, current_col, agg_op, inclusive_sum_result, num_group);
                res_col._null_mask.init_zero(_device, _stream, num_group, ShaderCollector<uint>::get_instance(_device)->set_shader);

                _stream << ShaderCollector<uint>::get_instance(_device)->reindex_bitmap_with_null_shader(res_col._null_mask, current_col._null_mask, inclusive_sum_result).dispatch(current_col.size());

                if (each_group_count.size() == 0 && (agg_op == AggeragateOp::MEAN || agg_op == AggeragateOp::COUNT)) {
                    each_group_count = unique_count(_device, _stream, adjacent_diff_result, inclusive_sum_result, num_group);
                }

                // print_buffer(_stream, each_group_count.view());

                if (agg_op == AggeragateOp::MEAN) {
                    type_dispatcher(current_col_type, sum_to_mean{}, _device, _stream, res_col, each_group_count.view().as<uint>());
                }

                // print_buffer(_stream, current_col._null_mask._data.view());
                if (agg_op == AggeragateOp::COUNT) {
                    res_col.load(_device, _stream, each_group_count);
                }


                res_columns.insert({new_col_name, std::move(res_col)});
            }
            _columns.erase(cur_kv);
        }

        _columns = std::move(res_columns);
        return this;

    }

    Table *sort(const luisa::string &name, SortOrder order) {
        if (_columns.find(name) == _columns.end()) return this;
        using namespace luisa;
        using namespace luisa::compute;

        Column &col = _columns[name];
        if (col.size() == 0) return this;

        const auto &type = col.dtype();
        Column sorted_result{type};
        auto indices = type_dispatcher(type, sort_column{}, _device, _stream, col, sorted_result, order);

        for (auto it = _columns.begin(); it != _columns.end(); ++it) {
            if (name == it->first) continue;
            Column &current_col = it->second;
            const auto &current_col_type = current_col.dtype();
            type_dispatcher(current_col_type, inverse_reindex{}, _device, _stream, current_col, indices);
        }

        _columns[name] = std::move(sorted_result);
        return this;
    }

    template <class T>
    Table *apply(const luisa::string &name, luisa::compute::Callable<T(T)> &apply_func) {
        if (_columns.find(name) == _columns.end()) return this;
        using namespace luisa;
        using namespace luisa::compute;

        Column &col = _columns[name];
        if (col.size() == 0) return this;
        const auto &type = col.dtype();

        bool is_valid = same_type<T>(type.id());

        if (!is_valid) {
            LUISA_WARNING("APPLY SKIP: encouter different type -- COL_TYPE: {} <==> FUNC_TYPE: {}", type_id_string(type.id()), typeid(T).name());
            return this;
        }
        apply_on_column_T{}.operator()<T>(_device, _stream, col, reinterpret_cast<void *>(&apply_func));
        
        return this;
    }

    template <class Ret, class T, std::enable_if_t<!std::is_same_v<Ret, T>, int> = 0>
    Table *apply(const luisa::string &name, luisa::compute::Callable<Ret(T)> &apply_func, TypeId ret_type_id = TypeId::EMPTY) {
        if (_columns.find(name) == _columns.end()) return this;
        using namespace luisa;
        using namespace luisa::compute;

        Column &col = _columns[name];
        if (col.size() == 0) return this;
        const auto &type = col.dtype();
        if (ret_type_id == TypeId::EMPTY) ret_type_id = type_to_id<Ret>();
        bool is_valid = same_type<T>(type.id()) && same_type<Ret>(ret_type_id);

        if (!is_valid) {
            LUISA_WARNING("APPLY SKIP: encouter different type -- COL_TYPE: {}({}) <==> FUNC_TYPE: {}({})", type_id_string(ret_type_id), type_id_string(type.id()), typeid(Ret).name(), typeid(T).name());
            return this;
        }

        apply_on_column_Ret_T{}.operator()<Ret, T>(_device, _stream, col, reinterpret_cast<void *>(&apply_func));
        col.set_dtype(ret_type_id);
        return this;
    }

    Table *join(Table &other, const luisa::string &col_left, const luisa::string &col_right, const JoinType &join_type = JoinType::LEFT) {
        if (_columns.find(col_left) == _columns.end() || other._columns.find(col_right) == other._columns.end()) {
            LUISA_WARNING("JOIN SKIP: column not found. left: {}, right: {}", col_left, col_right);
            return this;
        }
        using namespace luisa;
        using namespace luisa::compute;

        Column &left_col = _columns[col_left];
        Column &right_col = other._columns[col_right];

        if (left_col.dtype() != right_col.dtype()) {
            LUISA_WARNING("JOIN SKIP: column type not match. left: {}, right: {}", type_id_string(left_col.dtype().id()), type_id_string(right_col.dtype().id()));
            return this;
        }

        auto type = left_col.dtype().id();
        BufferIndex index_left, index_right;

        // Buffer<JoinIndex> join_result;
        if (join_type == JoinType::LEFT) {
            std::tie(index_left, index_right) = type_dispatcher(type, left_join{}, _device, _stream, left_col, right_col);
        } else if (join_type == JoinType::RIGHT) {
            std::tie(index_left, index_right) = type_dispatcher(type, right_join{}, _device, _stream, left_col, right_col);
        } else if (join_type == JoinType::INNER) {
            std::tie(index_left, index_right) = type_dispatcher(type, inner_join{}, _device, _stream, left_col, right_col);
        } else if (join_type == JoinType::OUTER) {
            std::tie(index_left, index_right) = type_dispatcher(type, outer_join{}, _device, _stream, left_col, right_col);
        }  else {
            LUISA_WARNING("Unsupported Join Type!");
            return this;
        }

        // print_buffer(_stream, index_left.view());
        // print_buffer(_stream, index_right.view());
        
        luisa::unordered_map<luisa::string, Column> join_result;
        if (join_type == JoinType::RIGHT) {
            fill_join_result(_device, _stream, index_right, other._columns, join_result);
            fill_join_result(_device, _stream, index_left, this->_columns, join_result);
        } else {
            fill_join_result(_device, _stream, index_left, this->_columns, join_result);
            fill_join_result(_device, _stream, index_right, other._columns, join_result);
        }
        _columns = std::move(join_result);
        return this;
    }

    Table *hashmap_join(Table &other, const luisa::string &col_left, const luisa::string &col_right, const JoinType &join_type = JoinType::LEFT) {
        if (_columns.find(col_left) == _columns.end() || other._columns.find(col_right) == other._columns.end()) {
            LUISA_WARNING("JOIN SKIP: column not found. left: {}, right: {}", col_left, col_right);
            return this;
        }
        using namespace luisa;
        using namespace luisa::compute;

        Column &left_col = _columns[col_left];
        Column &right_col = other._columns[col_right];

        if (left_col.dtype() != right_col.dtype()) {
            LUISA_WARNING("JOIN SKIP: column type not match. left: {}, right: {}", type_id_string(left_col.dtype().id()), type_id_string(right_col.dtype().id()));
            return this;
        }

        Clock clock;

        auto type = left_col.dtype().id();
        BufferIndex index_left, index_right;

        clock.tic();
        if (join_type == JoinType::LEFT) {
            std::tie(index_left, index_right) = type_dispatcher(type, hashmap_left_join{}, _device, _stream, left_col, right_col);
        } else if (join_type == JoinType::RIGHT) {
            std::tie(index_left, index_right) = type_dispatcher(type, hashmap_right_join{}, _device, _stream, left_col, right_col);
        } else if (join_type == JoinType::INNER) {
            std::tie(index_left, index_right) = type_dispatcher(type, inner_join{}, _device, _stream, left_col, right_col);
        } else if (join_type == JoinType::OUTER) {
            std::tie(index_left, index_right) = type_dispatcher(type, outer_join{}, _device, _stream, left_col, right_col);
        }  else {
            LUISA_WARNING("Unsupported Join Type!");
            return this;
        }
        LUISA_INFO("get left & right indices in {} ms", clock.toc());

        // print_buffer(_stream, index_left.view());
        // print_buffer(_stream, index_right.view());
        
        luisa::unordered_map<luisa::string, Column> join_result;
        if (join_type == JoinType::RIGHT) {
            fill_join_result(_device, _stream, index_right, other._columns, join_result);
            fill_join_result(_device, _stream, index_left, this->_columns, join_result);
        } else {
            fill_join_result(_device, _stream, index_left, this->_columns, join_result);
            fill_join_result(_device, _stream, index_right, other._columns, join_result);
        }
        _columns = std::move(join_result);
        return this;
    }

    void print_table() {
        printer.load(_device, _stream, _columns);
        printer.print(40);
    }

    void print_table_length() {
        printer.print_length(_columns);
    }
};
