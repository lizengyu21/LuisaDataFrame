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

    int stkcd_span = 1000;
    auto left = table.where("Stkcd", FilterOp::LESS_EQUAL, stkcd_span);
    auto right = table.where("Stkcd", FilterOp::GREATER_EQUAL, stkcd_span);
    right._where("Stkcd", FilterOp::LESS_EQUAL, int(stkcd_span * 2));

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = left.hashmap_join(right, "Trddt", "Trddt", JoinType::LEFT);
    }
    stream << synchronize();
    results["hashmap_join"]["test"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("hashmap join Stkcd in {} ms", clock.toc());

    clock.tic();
    for (int i = 0; i < test_round; ++i) {
        auto t = left.join(right, "Trddt", "Trddt", JoinType::LEFT);
    }
    stream << synchronize();
    results["naive_join"]["test"] = clock.toc() / static_cast<double>(test_round);
    LUISA_INFO("naive join Stkcd in {} ms", clock.toc());


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

    // Join
    json_ss << "    \"naive_join\": {\n";
    json_ss << "        \"test\": " << results["naive_join"]["test"] << "\n";
    json_ss << "    },\n";

    json_ss << "    \"hashmap_join\": {\n";
    json_ss << "        \"test\": " << results["hashmap_join"]["test"] << "\n";
    json_ss << "    }\n";
    json_ss << "}\n";



    // 写入文件
    std::ofstream file("./results/ablation_results.json");
    if (file.is_open()) {
        file << json_ss.str();
        file.close();
        std::cout << "Benchmark results saved to 'ours_benchmark_results.json'\n";
    } else {
        std::cerr << "Failed to open file 'ours_benchmark_results.json'\n";
    }

    return 0;
}