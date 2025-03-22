
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
#include <ludf/core/hashmap.h>
#include <ludf/util/hashmap/helper.h>

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

    Hashmap<uint> hm;
    init(device, stream, hm, 100);

    print_buffer(stream, hm._key.view().as<uint>());

    auto test_hash_shader = device.compile<1>([&](Var<Hashmap<uint>> hashmap){
        auto x = dispatch_x();
        auto t = hashmap->hash(x);
        device_log("{} -> {}", x, t);
    });

    stream << test_hash_shader(hm).dispatch(100);

    std::cout << "End.\n";
}