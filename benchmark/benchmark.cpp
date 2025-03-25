
// #include <ludf/core/type.h>
// #include <ludf/core/type_dispatcher.h>
// #include <ludf/column/column.h>
// #include <ludf/util/util.h>
// #include <ludf/table/table.h>
// #include <ludf/util/kernel.h>
// #include <ludf/io/read_csv.h>
// #include <random>
// #include <luisa/luisa-compute.h>
// #include <luisa/dsl/sugar.h>
// #include <luisa/dsl/syntax.h>

// using namespace luisa;
// using namespace luisa::compute;

// int main(int argc, char *argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <backend>\n";
//         return 1;
//     }

//     // Step 1.1: Create a context
//     Context context{argv[0]};

//     // Step 1.2: Load the CUDA backend plug-in and create a device
//     Device device = context.create_device(argv[1]);

//     // Step 2.1: Create a stream for command submission
//     Stream stream = device.create_stream();
//     Clock clock;

//     Table table(device, stream);

//     clock.tic();
//     unordered_map<string, TypeId> type = {
//         {"Stkcd", TypeId::INT32},
//         {"Trddt", TypeId::TIMESTAMP},
//         {"Opnprc", TypeId::FLOAT32},
//         {"Hiprc", TypeId::FLOAT32},
//         {"Loprc", TypeId::FLOAT32},
//         {"Clsprc", TypeId::FLOAT32}
//     };
//     read_csv("./data/TRD_Dalyr0.csv", table, type);
//     read_csv("./data/TRD_Dalyr1.csv", table, type);
//     read_csv("./data/TRD_Dalyr2.csv", table, type);
//     read_csv("./data/TRD_Dalyr3.csv", table, type);
//     read_csv("./data/TRD_Dalyr4.csv", table, type);
//     read_csv("./data/TRD_Dalyr5.csv", table, type);
//     LUISA_INFO("load csv data in {} ms", clock.toc());
//     int test_round = 100;

//     clock.tic();
//     for (int i = 0; i < test_round; ++i) {
//         auto t = table.where("Stkcd", FilterOp::LESS_EQUAL, 20000);
//     }
//     stream << synchronize();
//     LUISA_INFO("where Stkcd in {} ms", clock.toc());

//     clock.tic();
//     for (int i = 0; i < test_round; ++i) {
//         auto t = table.where("Clsprc", FilterOp::GREATER_EQUAL, 15.0f);
//     }
//     stream << synchronize();
//     LUISA_INFO("where Clsprc in {} ms", clock.toc());
    

//     Callable apply_func_Stkcd = [](Int a) {
//         return a * def(2);
//     };

//     Callable apply_func_Opnprc = [](Float a) {
//         return a * def(2.0f);
//     };



//     clock.tic();
//     for (int i = 0; i < test_round; ++i) {
//         auto t = table.apply("Stkcd", apply_func_Stkcd);
//     }
//     stream << synchronize();
//     LUISA_INFO("apply Stkcd in {} ms", clock.toc());

//     clock.tic();
//     for (int i = 0; i < test_round; ++i) {
//         auto t = table.apply("Opnprc", apply_func_Opnprc);
//     }
//     stream << synchronize();
//     LUISA_INFO("apply Opnprc in {} ms", clock.toc());

//     clock.tic();
//     for (int i = 0; i < test_round; ++i) {
//         auto t = table.sort("Stkcd", SortOrder::Ascending);
//     }
//     stream << synchronize();
//     LUISA_INFO("sort Stkcd in {} ms", clock.toc());

//     clock.tic();
//     for (int i = 0; i < test_round; ++i) {
//         auto t = table.sort("Clsprc", SortOrder::Ascending);
//     }
//     stream << synchronize();
//     LUISA_INFO("sort Clsprc in {} ms", clock.toc());

//     clock.tic();
//     for (int i = 0; i < test_round; ++i) {
//         auto t = table.sort("Hiprc", SortOrder::Ascending);
//     }
//     stream << synchronize();
//     LUISA_INFO("sort Hiprc in {} ms", clock.toc());

//     int stkcd_span = 1000;
//     auto left = table.where("Stkcd", FilterOp::LESS_EQUAL, stkcd_span);
//     auto right = table.where("Stkcd", FilterOp::GREATER_EQUAL, stkcd_span);
//     right._where("Stkcd", FilterOp::LESS_EQUAL, int(stkcd_span * 2));

//     // left.print_table_length();
//     // right.print_table_length();

//     clock.tic();
//     for (int i = 0; i < test_round; ++i) {
//         auto t = left.hashmap_join(right, "Trddt", "Trddt", JoinType::LEFT);
//     }
//     stream << synchronize();
//     LUISA_INFO("join Stkcd in {} ms", clock.toc());

//     auto agg_op_map = unordered_map<string, vector<AggeragateOp>>();

//     agg_op_map.insert({"Opnprc", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN}});
//     agg_op_map.insert({"Clsprc", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN}});
//     agg_op_map.insert({"Hiprc", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN}});
//     agg_op_map.insert({"Loprc", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN}});

//     clock.tic();
//     for (int i = 0; i < test_round; ++i) {
//         auto t = table.group_by("Stkcd", agg_op_map);
//     }
//     stream << synchronize();
//     LUISA_INFO("groupby Stkcd in {} ms", clock.toc());

//     std::cout << "End.\n";
// }

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
        {"Clsprc", TypeId::FLOAT32}
    };
    read_csv("./data/TRD_Dalyr0.csv", table, type);
    read_csv("./data/TRD_Dalyr1.csv", table, type);
    read_csv("./data/TRD_Dalyr2.csv", table, type);
    read_csv("./data/TRD_Dalyr3.csv", table, type);
    read_csv("./data/TRD_Dalyr4.csv", table, type);
    read_csv("./data/TRD_Dalyr5.csv", table, type);
    double load_time = clock.toc();
    LUISA_INFO("load csv data in {} ms", load_time);

    int test_round = 100;
    size_t data_size = table._columns.begin()->second.size();

    // 存储基准测试结果
    std::unordered_map<std::string, std::unordered_map<std::string, double>> results;
    std::stringstream json_ss;

    // Where 操作
    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.where("Stkcd", FilterOp::LESS_EQUAL, 20000);
    }
    stream << synchronize();
    results["where"]["Stkcd"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("where Stkcd in {} ms", clock.toc());

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.where("Clsprc", FilterOp::GREATER_EQUAL, 15.0f);
    }
    stream << synchronize();
    results["where"]["Clsprc"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("where Clsprc in {} ms", clock.toc());

    // Apply 操作
    Callable apply_func_Stkcd = [](Int a) { return a * def(2); };
    Callable apply_func_Opnprc = [](Float a) { return a * def(2.0f); };

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.apply("Stkcd", apply_func_Stkcd);
    }
    stream << synchronize();
    results["apply"]["Stkcd"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("apply Stkcd in {} ms", clock.toc());

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.apply("Opnprc", apply_func_Opnprc);
    }
    stream << synchronize();
    results["apply"]["Opnprc"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("apply Opnprc in {} ms", clock.toc());

    // Sort 操作
    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.sort("Stkcd", SortOrder::Ascending);
    }
    stream << synchronize();
    results["sort"]["Stkcd"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("sort Stkcd in {} ms", clock.toc());

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.sort("Clsprc", SortOrder::Ascending);
    }
    stream << synchronize();
    results["sort"]["Clsprc"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("sort Clsprc in {} ms", clock.toc());

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.sort("Hiprc", SortOrder::Ascending);
    }
    stream << synchronize();
    results["sort"]["Hiprc"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("sort Hiprc in {} ms", clock.toc());

    // Join 操作
    int stkcd_span = 1000;
    auto left = table.where("Stkcd", FilterOp::LESS_EQUAL, stkcd_span);
    auto right = table.where("Stkcd", FilterOp::GREATER_EQUAL, stkcd_span);
    right._where("Stkcd", FilterOp::LESS_EQUAL, int(stkcd_span * 2));

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = left.hashmap_join(right, "Trddt", "Trddt", JoinType::LEFT);
    }
    stream << synchronize();
    results["join"]["Stkcd_0_20_vs_20_40"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("join Stkcd in {} ms", clock.toc());

    // Groupby 操作
    auto agg_op_map = unordered_map<string, vector<AggeragateOp>>();
    agg_op_map.insert({"Opnprc", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN}});
    agg_op_map.insert({"Clsprc", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN}});
    agg_op_map.insert({"Hiprc", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN}});
    agg_op_map.insert({"Loprc", {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN}});

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.group_by("Stkcd", agg_op_map);
    }
    stream << synchronize();
    results["groupby"]["Stkcd_Clsprc"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("groupby Stkcd in {} ms", clock.toc());

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.interval("Stkcd", "", 60*60*24*30, {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN});
    }
    stream << synchronize();
    results["interval"]["30D"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("interval 30D in {} ms", clock.toc());

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.interval("Stkcd", "", 60*60*24*90, {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN});
    }
    stream << synchronize();
    results["interval"]["90D"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("interval 90D in {} ms", clock.toc());

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = table.interval("Stkcd", "", 60*60*24*365, {AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::MEAN});
    }
    stream << synchronize();
    results["interval"]["365D"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("interval 365D in {} ms", clock.toc());
    

    // 手动构建 JSON 字符串
    json_ss << "{\n";
    json_ss << "    \"metadata\": {\n";
    json_ss << "        \"test_rounds\": " << test_round << ",\n";
    json_ss << "        \"data_size\": " << data_size << ",\n";
    std::time_t now = std::time(nullptr);
    std::string timestamp = std::ctime(&now);
    timestamp.pop_back(); // 移除换行符
    json_ss << "        \"timestamp\": \"" << timestamp << "\"\n";
    json_ss << "    },\n";

    // Where
    json_ss << "    \"where\": {\n";
    json_ss << "        \"Stkcd\": " << std::fixed << std::setprecision(4) << results["where"]["Stkcd"] << ",\n";
    json_ss << "        \"Clsprc\": " << results["where"]["Clsprc"] << "\n";
    json_ss << "    },\n";

    // Apply
    json_ss << "    \"apply\": {\n";
    json_ss << "        \"Stkcd\": " << results["apply"]["Stkcd"] << ",\n";
    json_ss << "        \"Opnprc\": " << results["apply"]["Opnprc"] << "\n";
    json_ss << "    },\n";

    // Sort
    json_ss << "    \"sort\": {\n";
    json_ss << "        \"Stkcd\": " << results["sort"]["Stkcd"] << ",\n";
    json_ss << "        \"Clsprc\": " << results["sort"]["Clsprc"] << ",\n";
    json_ss << "        \"Hiprc\": " << results["sort"]["Hiprc"] << "\n";
    json_ss << "    },\n";

    // Join
    json_ss << "    \"join\": {\n";
    json_ss << "        \"Stkcd_0_20_vs_20_40\": " << results["join"]["Stkcd_0_20_vs_20_40"] << "\n";
    json_ss << "    },\n";

    // Interval
    json_ss << "    \"interval\": {\n";
    json_ss << "        \"30D\": " << results["interval"]["30D"] << ",\n";
    json_ss << "        \"90D\": " << results["interval"]["90D"] << ",\n";
    json_ss << "        \"365D\": " << results["interval"]["365D"] << "\n";
    json_ss << "    },\n";

    // Groupby
    json_ss << "    \"groupby\": {\n";
    json_ss << "        \"Stkcd_Clsprc\": " << results["groupby"]["Stkcd_Clsprc"] << "\n";
    json_ss << "    }\n";
    json_ss << "}\n";



    // 写入文件
    std::ofstream file("./results/ours_benchmark_results.json");
    if (file.is_open()) {
        file << json_ss.str();
        file.close();
        std::cout << "Benchmark results saved to 'ours_benchmark_results.json'\n";
    } else {
        std::cerr << "Failed to open file 'ours_benchmark_results.json'\n";
    }

    return 0;
}