
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

#include <ludf/core/bitmap.h>

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

    auto test = device.compile<1>([](Var<Bitmap> bitmap) {
        auto x = dispatch_x();
        auto y = bitmap->test(x);
        $if (!y) {
            device_log("Error init at index: {}", x);
        };
        bitmap->set(x);
        y = bitmap->test(x);
        $if (!y) {
            device_log("Error set and test failed at index: {}", x);
        };
        bitmap->clear(x);
        y = bitmap->test(x);
        $if (y) {
            device_log("Error clear and test failed at index: {}", x);
        };
    });

    uint size = std::stoi(argv[2]);
    Bitmap bitmap{
        device.create_buffer<uint>((size + 31) / 32),
        size
    };

    bitmap.init_one(device, stream, size);

    stream << test(bitmap).dispatch(size) << synchronize();

    std::cout << "End.\n";
}