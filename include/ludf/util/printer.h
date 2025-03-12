#pragma once
#include <ludf/table/table.h>
#include <luisa/luisa-compute.h>
#include <ludf/core/type_dispatcher.h>
#include <sstream>
#include <string>


struct Printer {
    luisa::unordered_map<luisa::string, void *> col_data;
    luisa::unordered_map<luisa::string, TypeId> col_type;
    luisa::unordered_map<luisa::string, luisa::vector<uint>> col_null_mask;
    size_t len;

    // struct alloc {
    //     template <class T>
    //     void *operator()(size_t size) {
    //         return new T[size];
    //     }
    // };

    void clear() {
        for (auto &entry : col_data) {
            if (entry.second) delete[] static_cast<char*>(entry.second);
        }
        col_data.clear();
        col_null_mask.clear();
        col_type.clear();
    }

    bool is_null(const luisa::string &col_name, size_t index) const {
        if (col_null_mask.find(col_name) == col_null_mask.end() || col_null_mask.at(col_name).empty()) {
            return false;
        }
        size_t word_index = index / 32;
        size_t bit_index = index % 32;
        return (col_null_mask.at(col_name)[word_index] & (1u << bit_index)) != 0;
    }

    void load(luisa::compute::Device &device, luisa::compute::Stream &stream, luisa::unordered_map<luisa::string, Column> &_columns) {
        clear();
        if (_columns.begin() == _columns.end()) return;
        len = _columns.begin()->second.size();
        for (auto &it : _columns) {
            if(it.second.size() != len) {
                LUISA_ERROR("FATAL ERROR: columns have different length");
            }
        }

        for (auto &it : _columns) {
            
            col_type[it.first] = it.second.dtype().id();
            col_null_mask[it.first] = luisa::vector<uint>(it.second._null_mask._data.size());
            auto size = len * id_to_size(it.second.dtype().id());

            void *data = new char[size];
            stream << it.second._data.copy_to(data);
            col_data[it.first] = data;
            if (it.second._null_mask._size != len && it.second._null_mask._size > 0) {
                LUISA_WARNING("null mask length NOT equal to data {} -- {}", it.second._null_mask._size, len);
            }
            if (it.second._null_mask._size > 0) {
                stream << it.second._null_mask._data.copy_to(col_null_mask[it.first].data());
            }
        }
        stream << luisa::compute::synchronize();
    }

    struct get_width {
        bool is_null(const luisa::vector<uint> &null_mask, size_t index) const {
            if (null_mask.empty()) {
                return false;
            }
            size_t word_index = index / 32;
            size_t bit_index = index % 32;
            return (null_mask[word_index] & (1u << bit_index)) != 0;
        }
    
        template <class T>
        void operator()(void *data, const luisa::vector<uint> &mask, size_t size, size_t &len) {
            T *ptr = reinterpret_cast<T*>(data);
            for (size_t i = 0; i < size; ++i) {
                if (is_null(mask, i)) len = std::max(len, 4ul);
                else {
                    if constexpr (std::is_same_v<T, float>) {
                        std::stringstream ss;
                        ss << std::fixed << std::setprecision(4) << ptr[i];
                        std::string formatted = ss.str();
                        len = std::max(len, formatted.length());
                    } else {
                        len = std::max(len, std::to_string(ptr[i]).length());
                    }
                }
            }
        }
    };

    void print(size_t max_rows = 20) {
        using namespace luisa;
        unordered_map<string, size_t> col_max_len;
        auto print_len = std::min(len, max_rows);
        bool overflow = max_rows < len;
        for (auto &it : col_type) {
            col_max_len[it.first] = std::max(it.first.length(), 4ul);
            type_dispatcher(it.second, get_width{}, col_data[it.first], col_null_mask[it.first], print_len, col_max_len[it.first]);
        }
        header(col_max_len);
        print_data(col_max_len, print_len, overflow);
        tail(len);
    }

    void tail(size_t total) {
        std::cout << "Total Rows: " << total << '\n';
    }

    void border(const luisa::unordered_map<luisa::string, size_t> &col_max_len) {
        for (auto it = col_max_len.cbegin(); it != col_max_len.cend(); ++it) {
            if (it == col_max_len.cbegin()) std::cout << "+";
            std::cout << std::string(it->second + 2, '-') << "+";
        }
        std::cout << "\n";
    }

    void header(const luisa::unordered_map<luisa::string, size_t> &col_max_len) {
        border(col_max_len);
        for (auto &it : col_max_len) {
            std::cout << "| " << std::setw(it.second) << std::left << it.first << " ";
        }
        std::cout << "|\n";
        border(col_max_len);
    }

    void print_data(const luisa::unordered_map<luisa::string, size_t> &col_max_len, size_t print_len, bool overflow) {
        for (size_t row = 0; row < print_len; ++row) {
            for (auto &col : col_type) {
                std::cout << "| ";
                if (is_null(col.first, row)) {
                    std::cout << std::setw(col_max_len.at(col.first)) << std::left << "NULL";
                } else {
                    type_dispatcher(col.second, print_value{}, col_data[col.first], row, col_max_len.at(col.first));
                }
                std::cout << " ";
            }
            std::cout << "|\n";
        }
        if (overflow) {
            for (auto &col : col_type) {
                std::cout << "| ";
                std::cout << std::setw(col_max_len.at(col.first)) << std::left << "...";
                std::cout << " ";
            }
            std::cout << "|\n";
        }
        border(col_max_len);
    }

    struct print_value {
        template <class T>
        void operator()(void *data, size_t row, size_t width) {
            T *ptr = reinterpret_cast<T*>(data);
            if constexpr (std::is_same_v<T, float>) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(4) << ptr[row];
                std::cout << std::setw(width) << std::left << ss.str();
            } else {
                std::cout << std::setw(width) << std::left << ptr[row];
            }
        }
    };
};