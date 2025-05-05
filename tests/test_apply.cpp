
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

    // 数据加载
    clock.tic();
    // unordered_map<string, TypeId> type = {
    //     {"Stkcd", TypeId::INT32},
    //     {"Trddt", TypeId::TIMESTAMP},
    //     {"Opnprc", TypeId::FLOAT32},
    //     {"Hiprc", TypeId::FLOAT32},
    //     {"Loprc", TypeId::FLOAT32},
    //     {"Clsprc", TypeId::FLOAT32}
    // };
    // read_csv("./data/TRD_Dalyr0.csv", table, type);
    double load_time = clock.toc();
    LUISA_INFO("load csv data in {} ms", load_time);


    vector<int> a = {0, 2, 4, 5};
    vector<float> b = {1.0, 2.1, 1, 5};

    unordered_map<string, std::pair<size_t, void *>> data;
    data["int"] = {a.size(), a.data()};
    data["float"] = {b.size(), b.data()};
    unordered_map<string, TypeId> type;
    type["int"] = TypeId::INT32;
    type["float"] = TypeId::FLOAT32;

    table.create_table(data, type);
    // table.create_column("int", TypeId::INT32, a);
    // table.create_column("float", TypeId::INT32, b);

    table.print_table();

    // auto &c = table["Trddt"];
    // std::cout << type_id_string(c.dtype().id()) << std::endl;





    // auto &lhs = table._columns["Clsprc"];
    // auto &rhs = table._columns["Hiprc"];

    // Callable apply_func = [](Float a, Float b) { return a + b + def(100.0f); };

    // table.apply(lhs, rhs, apply_func);

    std::cout << "End.\n";
}