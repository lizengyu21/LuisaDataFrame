#pragma once
#include <luisa/luisa-compute.h>
#include <ludf/column/column.h>
#include <ludf/core/type.h>
#include <ludf/util/util.h>

class Table {
    luisa::compute::Device &_device;
    luisa::compute::Stream &_stream;

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
        luisa::Clock clock;
        clock.tic();
        Table t{_device, _stream};
        for (auto it = _columns.cbegin(); it != _columns.cend(); ++it) {
            t.create_column(it->first, it->second.clone(_device, _stream));
        }
        LUISA_INFO("Query start in {} ms", clock.toc());
        return std::move(t);
    }

    void append_column(const luisa::string &name, void *data, size_t size) {
        Column &col = _columns[name];
        const auto &type = col.dtype();
        Column data_col{type};
        data_col.load(_device, _stream, data, size, true);
        type_dispatcher(type, concat_column{}, _device, _stream, col, data_col);
    }

    template <class T>
    void append_column(const luisa::string &name, luisa::vector<T> data) {
        append_column(name, data.data(), data.size_bytes());
    }

    void where(const luisa::string &name, const FilterOp op, void *threshold) {
        if (_columns.find(name) == _columns.end()) return;
        Column &col = _columns[name];
        const auto &type = col.dtype();
        auto reindex = type_dispatcher(type, make_inverse_reindex{}, _device, _stream, col, op, threshold);
        for (auto it = _columns.begin(); it != _columns.end(); ++it) {
            type_dispatcher(it->second.dtype(), inverse_reindex{}, _device, _stream, it->second, reindex);
        }
        // type_dispatcher(type, inverse_reindex{}, _device, _stream, col, reindex);
    }

    void group_by(const luisa::string &name, const luisa::unordered_map<luisa::string, luisa::vector<AggeragateOp>> &agg_op_map = {}) {
        using namespace luisa;
        using namespace luisa::compute;

        if (_columns.find(name) == _columns.end()) [[unlikely]] return;

        Column &col = _columns[name];
        if (col.size() == 0) return;

        const auto &type = col.dtype();
        Column sorted_result{type};
        auto indices = type_dispatcher(type, sort_column{}, _device, _stream, col, sorted_result);
        auto adjacent_diff_result = type_dispatcher(type, adjacent_diff{}, _device, _stream, sorted_result);
        auto inclusive_sum_result = inclusive_sum(_device, _stream, adjacent_diff_result);
        print_buffer(_stream, inclusive_sum_result.view());
        uint num_group;
        _stream << inclusive_sum_result.view(inclusive_sum_result.size() - 1, 1).copy_to(&num_group) << synchronize();
        ++num_group;

        type_dispatcher(type, print_column{}, _stream, col);
        
        type_dispatcher(type, inverse_reindex{}, _device, _stream, col, indices);
    
        type_dispatcher(type, print_column{}, _stream, col);

        type_dispatcher(type, reindex{}, _device, _stream, col, inclusive_sum_result, num_group);

        type_dispatcher(type, print_column{}, _stream, col);

        luisa::unordered_map<luisa::string, Column> res_columns;

        res_columns.insert({name, std::move(col)});

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
                auto res_col = type_dispatcher(current_col_type, aggregate_column{}, _device, _stream, current_col, agg_op, inclusive_sum_result, num_group);
                res_columns.insert({new_col_name, std::move(res_col)});
            }
        }

        _columns = std::move(res_columns);

    }

    void print_table() {
        for (auto it = _columns.begin(); it != _columns.end(); ++it) {
            std::cout << it->first << ": ";
            type_dispatcher(it->second.dtype(), print_column{}, _stream, it->second);
        }
    }
};