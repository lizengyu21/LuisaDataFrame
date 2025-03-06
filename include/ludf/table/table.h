#pragma once
#include <luisa/luisa-compute.h>
#include <ludf/column/column.h>
#include <ludf/core/type.h>
#include <ludf/util/util.h>

using namespace luisa;
using namespace luisa::compute;

class Table {
    Device &_device;
    Stream &_stream;

    void create_column(const string &name, Column &&col) {
        if (_columns.find(name) != _columns.end()) return;
        _columns.insert({name, std::move(col)});
    }

public:
    unordered_map<string, Column> _columns;

    Table(Device &device, Stream &stream) : _device(device), _stream(stream) {}

    void create_column(const string &name, DataType dtype) {
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

    void append_column(const string &name, void *data, size_t size) {
        Column &col = _columns[name];
        const auto &type = col.dtype();
        Column data_col{type};
        data_col.load(_device, _stream, data, size, true);
        type_dispatcher(type, concat_column{}, _device, _stream, col, data_col);
    }

    template <class T>
    void append_column(const string &name, vector<T> data) {
        append_column(name, data.data(), data.size_bytes());
    }

    void where(const string &name, const FilterOp op, void *threshold) {
        if (_columns.find(name) == _columns.end()) return;
        Column &col = _columns[name];
        const auto &type = col.dtype();
        auto reindex = type_dispatcher(type, make_inverse_reindex{}, _device, _stream, col, op, threshold);
        print_buffer(_stream, reindex.view());
    }
};