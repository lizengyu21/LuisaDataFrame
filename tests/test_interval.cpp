
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

    while (true) {
        int cd;
        // std::cin >> cd;
        uint span;
        std::cin >> span;
        int test_round;
        std::cin >> test_round;
        clock.tic();
        for (int i = 0; i < test_round; ++i) {
            table.query().interval("Stkcd", "", 60*60*24*span, {AggeragateOp::COUNT, AggeragateOp::MEAN, AggeragateOp::SUM, AggeragateOp::MAX, AggeragateOp::MIN});
        }
        LUISA_INFO("interval in {} ms for {} times, avg {} ms", clock.toc(), test_round, clock.toc() / double(test_round));
        // auto t = table.query();
        // // t.where("Stkcd", FilterOp::LESS_EQUAL, cd);
        // // t.where("Trddt", FilterOp::LESS_EQUAL, 1582588800u - 60*60*24*1u);

        // // t.sort("Stkcd", SortOrder::Ascending);
        // // t.print_table();
        // // t.sort("Trddt", SortOrder::Ascending);

        
        // t.print_table();
    }
    std::cout << "End.\n";
}