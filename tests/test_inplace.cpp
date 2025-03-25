
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
    LUISA_INFO("load csv data in {} ms", clock.toc());

    table.print_table();

    auto agg_op_map = unordered_map<string, vector<AggeragateOp>>();
    // agg_op_map.insert({"opnprc", {AggeragateOp::SUM, AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::COUNT, AggeragateOp::MEAN}});
    agg_op_map.insert({"Clsprc", {AggeragateOp::SUM, AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::COUNT, AggeragateOp::MEAN}});
    agg_op_map.insert({"Opnprc", {AggeragateOp::SUM, AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::COUNT, AggeragateOp::MEAN}});

    while (true) {
        std::string cmd, col;
        std::cin >> cmd;
        clock.tic();
        if (cmd == "where") {
            int th;
            std::cin >> th;
            clock.tic();
            auto t = std::move(table.where("Stkcd", FilterOp::LESS_EQUAL, th));
            // t.print_table();
        } else if (cmd == "sort") {
            std::string name;
            std::cin >> name;
            clock.tic();
            auto t = std::move(table.sort(name.c_str(), SortOrder::Ascending));
            // t.print_table();
        } else if (cmd == "join") {
            auto t1 = table.where("Stkcd", FilterOp::LESS_EQUAL, 4);
            auto t2 = table.where("Stkcd", FilterOp::GREATER_EQUAL, 2);
            t2._where("Stkcd", FilterOp::LESS_EQUAL, 5);
            
            clock.tic();
            auto t3 = t1.hashmap_join(t2, "Stkcd", "Stkcd", JoinType::LEFT);
            // t3.print_table();
        }
        stream << synchronize();
        LUISA_INFO("Op {} in {} ms", cmd, clock.toc());
    }
    
    std::cout << "End.\n";
}