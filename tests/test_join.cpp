
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
    table.create_column("id", TypeId::INT32);
    table.append_column("id", vector<int>{0, 1, 2, 3});
    table.create_column("opnprc", TypeId::INT32);
    table.append_column("opnprc", vector<int>{0, 1, 2, 3});

    Table table2(device, stream);
    table2.create_column("id2", TypeId::INT32);
    table2.append_column("id2", vector<int>{1, 2, 3, 4});
    table2.create_column("clsprc", TypeId::FLOAT32);
    table2.append_column("clsprc", vector<float>{1.5, 2.5, 3.5, 4.5});


    table.join(table2, "id", "id2");

    std::cout << "End.\n";
}