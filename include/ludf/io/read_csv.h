#pragma once
#include <luisa/luisa-compute.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ludf/core/type.h>
#include <ludf/table/table.h>
#include <parser.hpp>

inline void remove_BOM(std::ifstream& file) {
    const char BOM[3] = { (char)0xEF, (char)0xBB, (char)0xBF };
    char buffer[3];

    // 如果文件以 BOM 开头，移除 BOM
    file.read(buffer, 3);
    if (buffer[0] == BOM[0] && buffer[1] == BOM[1] && buffer[2] == BOM[2]) {
        ;
    } else {
        file.seekg(0, std::ios::beg); // 如果没有BOM，回到文件开始位置
    }
}

inline void parse_csv(auto &file, auto &data) {
    remove_BOM(file);
    luisa::string line;
    bool first_line = true;
    luisa::vector<luisa::string> headers;

    int i = 10;
    while (std::getline(file, line)) {
        std::stringstream ss(line.c_str());
        luisa::string cell;
        size_t colIndex = 0;


        while (std::getline(ss, cell, ',')) {
            cell.erase(cell.begin(), std::find_if(cell.begin(), cell.end(), [](unsigned char ch) {
                return !std::isspace(ch);  // 去除开头的空格
            }));

            cell.erase(std::find_if(cell.rbegin(), cell.rend(), [](unsigned char ch) {
                return !std::isspace(ch);  // 去除结尾的空格
            }).base(), cell.end());

            if (!cell.empty() && cell.front() == '"' && cell.back() == '"') {
                cell = cell.substr(1, cell.size() - 2); // 移除双引号
            }

            if (first_line) {
                headers.push_back(cell);  // 保存标题行
            } else {
                data[headers[colIndex]].push_back(cell);  // 处理数据
            }
            colIndex++;
        }

        first_line = false;
        // if (i-- == 0) break;
    }
}

inline uint32_t parse_time_to_timestamp(const std::string& s) {
    std::tm tm = {};
    std::istringstream ss(s);
    // 首先尝试常见的日期时间格式：YYYY-MM-DD HH:MM:SS
    ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    if (!ss.fail()) {
        std::time_t time = std::mktime(&tm);
        if (time != -1) {
            return static_cast<uint32_t>(time);  // 返回时间戳
        }
    }

    // 如果不匹配，尝试仅有日期的格式：YYYY-MM-DD
    ss.clear();  // 清除之前的错误状态
    ss.str(s);
    ss >> std::get_time(&tm, "%Y-%m-%d");
    if (!ss.fail()) {
        std::time_t time = std::mktime(&tm);
        if (time != -1) {
            return static_cast<uint32_t>(time);  // 返回时间戳
        }
    }

    // 如果没有匹配任何格式，抛出错误
    throw std::runtime_error("Unable to parse time format: " + s);
}

inline void parse_type(auto &data, auto &type) {
    for (const auto& column : data) {
        const luisa::string &column_name = column.first;
        const luisa::vector<luisa::string> &values = column.second;

        bool has_time = false;
        bool has_float = false;
        bool real_float = false;

        bool has_int = false;
        bool all_int = true;

        // 检测列数据类型：优先检查时间戳 > float > int
        for (const auto& value : values) {
            try {
                parse_time_to_timestamp(value.c_str());  // 尝试将值转换为时间戳
                has_time = true;
            } catch (...) {

            }

            try {
                std::stof(value.c_str());  // 如果是浮动数
                has_float = true;
                if (value.find('.') != luisa::string::npos) {
                    real_float = true;
                }
            } catch (...) {
                
            }

            try {
                auto t = std::stoi(value.c_str());  // 如果是整数
                has_int = true;
            } catch (...) {
                all_int = false;
            }
        }

        if (has_time) {
            type[column_name] = TypeId::TIMESTAMP;
            std::cout << column_name << " detected as TIMESTAMP\n";
        } else if (all_int && !real_float) {
            type[column_name] = TypeId::INT32;
            std::cout << column_name << " detected as INT32\n";
        } else if (has_float) {
            type[column_name] = TypeId::FLOAT32;
            std::cout << column_name << " detected as FLOAT32\n";
        } else {
            LUISA_ERROR("Currently do not support string type for column: {}", column_name);
        }
    }
}

// inline void convert_store_column(luisa::vector<luisa::string> &raw_data) {
//     for (const auto &cell : raw_data) {
//         ;
//     }
// }

inline void convert_data(Table &table, luisa::unordered_map<luisa::string, luisa::vector<luisa::string>> &data, luisa::unordered_map<luisa::string, TypeId> &type) {
    for (const auto &pair : data) {
        const luisa::string &column_name = pair.first;
        const luisa::vector<luisa::string> &values = pair.second;
        const TypeId &column_type = type[column_name];

        if (column_type == TypeId::TIMESTAMP) {
            luisa::vector<uint> col_values;
            for (const auto &value : values) {
                col_values.push_back(parse_time_to_timestamp(value.c_str()));
            }
            table.append_column(column_name, col_values);
        } else if (column_type == TypeId::INT32) {
            luisa::vector<int> col_values;
            for (const auto &value : values) {
                col_values.push_back(std::stoi(value.c_str()));
            }
            table.append_column(column_name, col_values);
        } else if (column_type == TypeId::FLOAT32) {
            luisa::vector<float> col_values;
            for (const auto &value : values) {
                col_values.push_back(std::stof(value.c_str()));
            }
            table.append_column(column_name, col_values);
        } else {
            LUISA_ERROR("Unsupported type!");
        }
    }
}

// inline void read_csv(const luisa::string &filename, Table &table) {
//     std::ifstream file(filename.c_str());
//     if (!file.is_open()) {
//         LUISA_WARNING("Cannot open file {}", filename);
//         return;
//     }

//     luisa::unordered_map<luisa::string, luisa::vector<luisa::string>> data;
//     luisa::unordered_map<luisa::string, TypeId> type;
//     parse_csv(file, data);
//     parse_type(data, type);

//     for (auto &pair : type) {
//         table.create_column(pair.first, pair.second);
//     }

//     convert_data(table, data, type);
// }

inline void read_csv(const luisa::string &filename, Table &table, luisa::unordered_map<luisa::string, TypeId> type) {
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        LUISA_WARNING("Cannot open file {}", filename);
        return;
    }
    using namespace aria::csv;
    CsvParser parser(file);

    luisa::vector<luisa::string> col_name;
    luisa::unordered_map<luisa::string, luisa::vector<int>> col_data;

    bool first_row = true;

    for (auto& row : parser) {
        if (first_row) {
            size_t col_count= row.size();
            if (type.size() != col_count) {
                LUISA_WARNING("csv with dismatched col size.");
                return;
            }
            for (auto& field : row) {
                if (type.find(field.c_str()) == type.end()) {
                    LUISA_WARNING("csv with dismatched col name.");
                    return;
                }
                col_name.push_back(field.c_str());
            }
            first_row = false;
            continue;
        }

        int i = 0;
        for (auto& field : row) {
            auto name = col_name[i++];
            auto t = type[name];
            if (t == TypeId::TIMESTAMP) {
                auto ts = parse_time_to_timestamp(field);
                col_data[name].push_back(*reinterpret_cast<int*>(&ts));
            } else if (t == TypeId::INT32) {
                col_data[name].push_back(std::stoi(field));
            } else if (t == TypeId::FLOAT32) {
                float fdata = std::stof(field);
                col_data[name].push_back(*reinterpret_cast<int*>(&fdata));
            } else if (t == TypeId::UINT32) {
                uint udata = std::stoi(field);
                col_data[name].push_back(*reinterpret_cast<int*>(&udata));
            } else {
                LUISA_WARNING("Unsupported type.");
                return;
            }
        } 
    }

    for (auto &name : col_name) {
        table.create_column(name, type[name]);
        table.append_column(name, col_data[name]);
    }
}

