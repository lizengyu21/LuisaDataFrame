#include <ludf/core/type.h>
#include <ludf/core/type_dispatcher.h>
#include <ludf/column/column.h>
#include <ludf/util/util.h>
#include <ludf/table/table.h>
#include <ludf/util/kernel.h>

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
    vector<int> t = {0, 1, 2, 3};
    vector<int> t2 = {0, 1, 2, 3};

    table.append_column("id", t);
    table.append_column("id", t);
    table.append_column("id", t);

    // auto tab = table.query();

    print_buffer(stream, table._columns["id"].view<int>());

    int th = std::stoi(argv[2]);
    table.where("id", FilterOp::LESS, &th);

    // print_buffer(stream, tab._columns["id"].view<int>());

    // print_buffer(stream, res.view().as<int>());
    std::cout << "End.\n";
}