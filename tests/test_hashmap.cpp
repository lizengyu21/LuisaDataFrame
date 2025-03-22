
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
    uint capacity, test_size, mod;
    std::cin >> capacity >> test_size >> mod;
    hashmap_init(device, stream, hm, capacity);

    std::cout << "Input: " << capacity << " " << test_size << " " << mod << std::endl;

    auto test_hash_shader = device.compile<1>([&](Var<Hashmap<uint>> hashmap){
        auto x = dispatch_x();
        auto insert_key = x % mod;
        hashmap->insert(insert_key);
    });

    auto count_buffer = device.create_buffer<uint>(mod);

    auto test_find_shader = device.compile<1>([&](Var<Hashmap<uint>> hashmap){
        auto x = dispatch_x();
        auto counter = hashmap->find(x);
        count_buffer->write(x, counter);
    });

    stream << test_hash_shader(hm).dispatch(test_size) << synchronize();

    print_buffer(stream, hm._key.view().as<uint>());
    print_buffer(stream, hm._counter.view());

    stream << test_find_shader(hm).dispatch(mod) << synchronize();

    print_buffer(stream, count_buffer.view());

    std::cout << "End.\n";
}