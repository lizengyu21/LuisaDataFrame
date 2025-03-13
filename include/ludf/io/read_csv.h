#pragma once
#include <luisa/luisa-compute.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ludf/core/type.h>
#include <ludf/table/table.h>

inline void remove_BOM(std::ifstream& file) {
    const char BOM[3] = { (char)0xEF, (char)0xBB, (char)0xBF };
    char buffer[3];

    // 如果文件以 BOM 开头，移除 BOM
    file.read(buffer, 3);
    if (buffer[0] == BOM[0] && buffer[1] == BOM[1] && buffer[2] == BOM[2]) {
        std::cout << "BOM detected and removed.\n";
    } else {
        file.seekg(0, std::ios::beg); // 如果没有BOM，回到文件开始位置
    }
}

inline void parse_csv(auto &file, auto &data) {
    remove_BOM(file);
    std::string line;
    bool first_line = true;
    std::vector<std::string> headers;

    int i = 10;
    while (std::getline(file, line)) {
        std::cout << line << '\n';
        std::stringstream ss(line);
        std::string cell;
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
        if (i-- == 0) break;
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
        const std::string &column_name = column.first;
        const std::vector<std::string> &values = column.second;

        bool has_time = false;
        bool has_float = false;
        bool has_int = true;

        // 检测列数据类型：优先检查时间戳 > float > int
        for (const auto& value : values) {
            try {
                parse_time_to_timestamp(value);  // 尝试将值转换为时间戳
                has_time = true;
                break;  // 一旦发现时间戳，标记列为时间戳类型
            } catch (...) {
                // 无需做任何事情，继续检查其他类型
            }

            try {
                std::stof(value);  // 如果是浮动数
                has_float = true;
            } catch (...) {
                // 无需做任何事情，继续检查其他类型
            }

            try {
                std::stoi(value);  // 如果是整数
            } catch (...) {
                has_int = false;  // 如果不能转换为整数，则不是整数类型
            }
        }
        if (has_time) {
            type[column_name] = TypeId::TIMESTAMP;
            std::cout << "Column \"" << column_name << "\" is detected as TIMESTAMP." << std::endl;
        } else if (has_float) {
            type[column_name] = TypeId::FLOAT32;
            std::cout << "Column \"" << column_name << "\" is detected as FLOAT32." << std::endl;
        } else if (has_int) {
            type[column_name] = TypeId::INT32;
            std::cout << "Column \"" << column_name << "\" is detected as INT32." << std::endl;
        } else {
            LUISA_ERROR("Currently do not support string type for column: {}", column_name);
        }
    }
}

inline void convert_store_column(std::vector<std::string> &raw_data) {
    for (const auto &cell : raw_data) {
        ;
    }
}

inline void read_csv(const std::string &filename, Table &table) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        LUISA_WARNING("Cannot open file {}", filename);
        return;
    }

    std::unordered_map<std::string, std::vector<std::string>> data;
    std::unordered_map<std::string, TypeId> type;
    parse_csv(file, data);
    parse_type(data, type);
}

