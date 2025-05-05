#include <ludf/core/type.h>
#include <ludf/core/type_dispatcher.h>
#include <ludf/column/column.h>
#include <ludf/util/util.h>
#include <ludf/table/table.h>
#include <ludf/util/kernel.h>
#include <ludf/io/read_csv.h>
#include <random>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

// convert std::unordered_map<std::string, std::unordered_map<std::string, double>> result to json string
std::stringstream convert_to_json(const std::unordered_map<std::string, std::unordered_map<std::string, double>> &results, const int &test_rounds, const int &data_size) {
    std::stringstream json_ss;
    json_ss << "{";
    // print meta data
    json_ss << "\n    \"metadata\": {";
    json_ss << "\n        \"test_rounds\": " << test_rounds << ",";
    json_ss << "\n        \"data_size\": " << data_size << ",";
    json_ss << "\n        \"timestamp\": \"" << std::time(nullptr) << "\"";
    json_ss << "\n    },";
    // print results
    for (const auto &outer_pair : results) {
        json_ss << "\n    \"" << outer_pair.first << "\": {";
        for (const auto &inner_pair : outer_pair.second) {
            json_ss << "\n        \"" << inner_pair.first << "\": " << inner_pair.second << ",";
        }
        json_ss.seekp(-1, json_ss.cur); // remove last comma
        json_ss << "\n    },";
    }
    json_ss.seekp(-1, json_ss.cur); // remove last comma
    json_ss << "\n}";
    return json_ss;
}

using namespace luisa;
using namespace luisa::compute;

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

    // 数据加载
    clock.tic();

    unordered_map<string, TypeId> type{{"timestamp", TypeId::TIMESTAMP}};

    int int_count = 4;
    int float_count = 4;

    for (int i = 1; i <= int_count; ++i) {
        type.insert({("int_" + std::to_string(i)).c_str(), TypeId::INT32});
    }
    for (int i = 1; i <= float_count; ++i) {
        type.insert({("float_" + std::to_string(i)).c_str(), TypeId::FLOAT32});
    }

    // LUISA_INFO("type count: {}", type.size());

    read_csv("./data/synthesis_data.csv", table, type);
    double load_time = clock.toc();
    LUISA_INFO("load csv data in {} ms", load_time);
    table.print_table_length();

    int test_round = 100;
    size_t data_size = table._columns.begin()->second.size();

    // 存储基准测试结果
    std::unordered_map<std::string, std::unordered_map<std::string, double>> results;

    // Where 操作
    for (int i = 1; i <= int_count; ++i) {
        clock.tic();
        auto col_name = ("int_" + std::to_string(i)).c_str();
        for (int _ = 0; _ < test_round; ++_) {
            auto t = table.where(col_name, FilterOp::LESS_EQUAL, 100);
        }
        stream << synchronize();
        results["where"][col_name] = clock.toc() / static_cast<double>(test_round);
        LUISA_INFO("where int_{} in {} ms, average {} ms", i, clock.toc(), clock.toc() / static_cast<double>(test_round));
    }

    for (int i = 1; i <= float_count; ++i) {
        clock.tic();
        auto col_name = ("float_" + std::to_string(i)).c_str();
        for (int _ = 0; _ < test_round; ++_) {
            auto t = table.where(col_name, FilterOp::GREATER, -55.0f);
        }
        stream << synchronize();
        results["where"][col_name] = clock.toc() / static_cast<double>(test_round);
        LUISA_INFO("where float_{} in {} ms, average {} ms", i, clock.toc(), clock.toc() / static_cast<double>(test_round));
    }

    // Apply 操作
    Callable apply_func_int = [](Int a) { return a * def(2); };
    for (int i = 1; i <= int_count; ++i) {
        clock.tic();
        auto col_name = ("int_" + std::to_string(i)).c_str();
        for (int _ = 0; _ < test_round; ++_) {
            auto t = table.apply(col_name, apply_func_int);
        }
        stream << synchronize();
        results["apply"][col_name] = clock.toc() / static_cast<double>(test_round);
        LUISA_INFO("apply int_{} in {} ms, average {} ms", i, clock.toc(), clock.toc() / static_cast<double>(test_round));
    }
    Callable apply_func_float = [](Float a) { return a * def(2.0f); };
    for (int i = 1; i <= float_count; ++i) {
        clock.tic();
        auto col_name = ("float_" + std::to_string(i)).c_str();
        for (int _ = 0; _ < test_round; ++_) {
            auto t = table.apply(col_name, apply_func_float);
        }
        stream << synchronize();
        results["apply"][col_name] = clock.toc() / static_cast<double>(test_round);
        LUISA_INFO("apply float_{} in {} ms, average {} ms", i, clock.toc(), clock.toc() / static_cast<double>(test_round));
    }

    // Sort 操作
    for (int i = 1; i <= int_count; ++i) {
        clock.tic();
        auto col_name = ("int_" + std::to_string(i)).c_str();
        for (int _ = 0; _ < test_round; ++_) {
            auto t = table.sort(col_name, SortOrder::Ascending);
        }
        stream << synchronize();
        results["sort"][col_name] = clock.toc() / static_cast<double>(test_round);
        LUISA_INFO("sort int_{} in {} ms, average {} ms", i, clock.toc(), clock.toc() / static_cast<double>(test_round));
    }

    for (int i = 1; i <= float_count; ++i) {
        clock.tic();
        auto col_name = ("float_" + std::to_string(i)).c_str();
        for (int _ = 0; _ < test_round; ++_) {
            auto t = table.sort(col_name, SortOrder::Ascending);
        }
        stream << synchronize();
        results["sort"][col_name] = clock.toc() / static_cast<double>(test_round);
        LUISA_INFO("sort float_{} in {} ms, average {} ms", i, clock.toc(), clock.toc() / static_cast<double>(test_round));
    }

    // Join 操作
    int int_1_span = 7;
    auto left = table.where("int_1", FilterOp::LESS_EQUAL, int_1_span);
    left._where("int_1", FilterOp::GREATER_EQUAL, 0);

    auto right = table.where("int_1", FilterOp::GREATER_EQUAL, int_1_span);
    right._where("int_1", FilterOp::LESS_EQUAL, int_1_span * 2);

    stream << synchronize();

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = left.hashmap_join(right, "int_2", "int_2", JoinType::LEFT);
    }
    stream << synchronize();
    results["join"]["int_1"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("join int_1 in {} ms, average {} ms", clock.toc(), clock.toc() / static_cast<double>(test_round));

    // Groupby 操作
    auto agg_op_map = unordered_map<string, vector<AggeragateOp>>();
    // agg_op_map.insert({"int_1", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN, AggeragateOp::SUM}});
    agg_op_map.insert({"int_2", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN, AggeragateOp::SUM}});
    agg_op_map.insert({"int_3", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN, AggeragateOp::SUM}});
    agg_op_map.insert({"int_4", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN, AggeragateOp::SUM}});
    agg_op_map.insert({"float_1", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN, AggeragateOp::SUM}});
    agg_op_map.insert({"float_2", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN, AggeragateOp::SUM}});
    agg_op_map.insert({"float_3", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN, AggeragateOp::SUM}});
    agg_op_map.insert({"float_4", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN, AggeragateOp::SUM}});

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.group_by("int_1", agg_op_map);
    }
    stream << synchronize();
    results["groupby"]["int_1"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("groupby int_1 in {} ms, average {} ms", clock.toc(), clock.toc() / static_cast<double>(test_round));

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.interval("int_2", "", 60*60, {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN, AggeragateOp::SUM, AggeragateOp::COUNT});
    }
    stream << synchronize();
    results["interval"]["1H"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("interval 1H in {} ms, average {} ms", clock.toc(), clock.toc() / static_cast<double>(test_round));
    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.interval("int_2", "", 60*60*24, {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN, AggeragateOp::SUM, AggeragateOp::COUNT});
    }
    stream << synchronize();
    results["interval"]["1D"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("interval 1D in {} ms, average {} ms", clock.toc(), clock.toc() / static_cast<double>(test_round));
    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.interval("int_2", "", 60*60*24*30, {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN, AggeragateOp::SUM, AggeragateOp::COUNT});
    }
    stream << synchronize();
    results["interval"]["1M"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("interval 1M in {} ms, average {} ms", clock.toc(), clock.toc() / static_cast<double>(test_round));
    
    auto json_ss = convert_to_json(results, test_round, data_size);
    // LUISA_INFO("results json: {}", json_ss.str());
    
    // results["interval"]["30D"] = clock.toc() / static_cast<double>(test_round);
    // LUISA_INFO("interval 30D in {} ms", clock.toc());

    // clock.tic();
    // for (int i = 0; i < test_round; ++i) {
    //     auto t = table.interval("Stkcd", "", 60*60*24*90, {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN});
    // }
    // stream << synchronize();
    // results["interval"]["90D"] = clock.toc() / static_cast<double>(test_round);
    // LUISA_INFO("interval 90D in {} ms", clock.toc());

    // clock.tic();
    // for (int i = 0; i < test_round; ++i) {
    //     auto t = table.interval("Stkcd", "", 60*60*24*365, {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN});
    // }
    // stream << synchronize();
    // results["interval"]["365D"] = clock.toc() / static_cast<double>(test_round);
    // LUISA_INFO("interval 365D in {} ms", clock.toc());
    

    // // 手动构建 JSON 字符串
    // json_ss << "{\n";
    // json_ss << "    \"metadata\": {\n";
    // json_ss << "        \"test_rounds\": " << test_round << ",\n";
    // json_ss << "        \"data_size\": " << data_size << ",\n";
    // std::time_t now = std::time(nullptr);
    // std::string timestamp = std::ctime(&now);
    // timestamp.pop_back(); // 移除换行符
    // json_ss << "        \"timestamp\": \"" << timestamp << "\"\n";
    // json_ss << "    },\n";

    // // Where
    // json_ss << "    \"where\": {\n";
    // json_ss << "        \"Stkcd\": " << std::fixed << std::setprecision(4) << results["where"]["Stkcd"] << ",\n";
    // json_ss << "        \"Clsprc\": " << results["where"]["Clsprc"] << "\n";
    // json_ss << "    },\n";

    // // Apply
    // json_ss << "    \"apply\": {\n";
    // json_ss << "        \"Stkcd\": " << results["apply"]["Stkcd"] << ",\n";
    // json_ss << "        \"Opnprc\": " << results["apply"]["Opnprc"] << "\n";
    // json_ss << "    },\n";

    // // Sort
    // json_ss << "    \"sort\": {\n";
    // json_ss << "        \"Stkcd\": " << results["sort"]["Stkcd"] << ",\n";
    // json_ss << "        \"Clsprc\": " << results["sort"]["Clsprc"] << ",\n";
    // json_ss << "        \"Hiprc\": " << results["sort"]["Hiprc"] << "\n";
    // json_ss << "    },\n";

    // // Join
    // json_ss << "    \"join\": {\n";
    // json_ss << "        \"Stkcd_0_20_vs_20_40\": " << results["join"]["Stkcd_0_20_vs_20_40"] << "\n";
    // json_ss << "    },\n";

    // // Interval
    // json_ss << "    \"interval\": {\n";
    // json_ss << "        \"30D\": " << results["interval"]["30D"] << ",\n";
    // json_ss << "        \"90D\": " << results["interval"]["90D"] << ",\n";
    // json_ss << "        \"365D\": " << results["interval"]["365D"] << "\n";
    // json_ss << "    },\n";

    // // Groupby
    // json_ss << "    \"groupby\": {\n";
    // json_ss << "        \"Stkcd_Clsprc\": " << results["groupby"]["Stkcd_Clsprc"] << "\n";
    // json_ss << "    }\n";
    // json_ss << "}\n";



    // 写入文件
    std::string json_filename = "./results/ours_syn_benchmark_results.json";

    std::ofstream file(json_filename);
    if (file.is_open()) {
        file << json_ss.str();
        file.close();
        std::cout << "Benchmark results saved to " << json_filename << "\n";
    } else {
        std::cerr << "Error opening file for writing: " << json_filename << "\n";
    }

    return 0;
}