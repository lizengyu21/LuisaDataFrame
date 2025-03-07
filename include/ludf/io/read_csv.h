#pragma once
#include <luisa/luisa-compute.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <ludf/core/type.h>
#include <ludf/table/table.h>

using namespace luisa;
using namespace luisa::compute;

inline void read_csv(const string &filename, Table &table) {
    std::ifstream file(filename.c_str());
    if (!file.is_open()) {
        std::cerr << "Error: file not found." << std::endl;
        return;
    }
    string line;
    std::getline(file, line);
    std::stringstream ss(line);
    string token;
    vector<string> columns;
    while (std::getline(ss, token, ',')) {
        // remove leading and trailing spaces

        // remove leading and trailing quatation marks
        if (token[0] == '\"') {
            token = token.substr(1, token.size() - 2);
        }
        columns.push_back(token);
    }
    
    for (auto &col : columns) {
        std::cout << col << std::endl;
    }

    // parse types from the data string
    std::getline(file, line);
    ss = std::stringstream(line);
    vector<DataType> types;


}