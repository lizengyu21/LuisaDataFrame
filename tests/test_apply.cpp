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
    unordered_map<string, TypeId> type = {
        {"Stkcd", TypeId::INT32},
        {"Trddt", TypeId::TIMESTAMP},
        {"Opnprc", TypeId::FLOAT32},
        {"Hiprc", TypeId::FLOAT32},
        {"Loprc", TypeId::FLOAT32},
        {"Clsprc", TypeId::FLOAT32},
        {"PrevClsprc", TypeId::FLOAT32},
    };
    read_csv("./data/TRD_Dalyr_with_PrevClsprc.csv", table, type);
    double load_time = clock.toc();
    LUISA_INFO("load csv data in {} ms", load_time);

    int test_round = 100;
    size_t data_size = table._columns.begin()->second.size();

    // 存储基准测试结果
    std::unordered_map<std::string, std::unordered_map<std::string, double>> results;
    Callable apply_func = [](Float cls, Float prev_cls) {
        return (cls - prev_cls) / prev_cls * 100.0f;
    };
    auto agg_op_map = unordered_map<string, vector<AggeragateOp>>();
    agg_op_map.insert({"DailyReturn", {AggeragateOp::MEAN, AggeragateOp::MAX}});
    agg_op_map.insert({"Trddt", {AggeragateOp::COUNT}});


    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto res = table.apply("Clsprc", "PrevClsprc", apply_func);
        table["DailyReturn"] = std::move(res);
        auto filtered_table = table.where("DailyReturn", FilterOp::GREATER, 2.0f);

        auto agg_table = filtered_table.group_by("Stkcd", agg_op_map);
        agg_table._sort("MEAN(DailyReturn)", SortOrder::Descending);
        // agg_table.print_table();
    }
    stream << synchronize();
    results["Result"]["DailyReturn"] = clock.toc() / static_cast<double>(test_round);

    table.erase("DailyReturn");
    agg_op_map.clear();
    agg_op_map.insert({"WeeklyReturn", {AggeragateOp::MEAN, AggeragateOp::MAX, AggeragateOp::COUNT}});
    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto week_data = table.interval("Stkcd", "", 60*60*24*7, {AggeragateOp::MEAN});
        auto res = week_data.apply("MEAN(Clsprc)", "MEAN(PrevClsprc)", apply_func);
        week_data["WeeklyReturn"] = std::move(res);
        auto filtered_table = week_data.where("WeeklyReturn", FilterOp::GREATER, 2.0f);
        auto agg_table = filtered_table.group_by("Stkcd", agg_op_map);
        agg_table._sort("WeeklyReturn", SortOrder::Descending);
        // agg_table.print_table();
    }
    stream << synchronize();
    results["Result"]["WeeklyReturn"] = clock.toc() / static_cast<double>(test_round);

    auto json_ss = convert_to_json(results, test_round, data_size);

    // 写入文件
    std::string json_filename = "./test.json";

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