
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
// using namespace luisa::compute::cuda::lcub;

template <typename T>
vector<T> createRandomVector(size_t size, T minValue, T maxValue, unsigned int seed) {
    // 创建一个随机数生成器，并用种子初始化
    std::mt19937 gen(seed);

    // 根据 T 的类型选择合适的随机数分布
    if constexpr (std::is_integral<T>::value) {
        // 对于整数类型，使用均匀整数分布
        std::uniform_int_distribution<T> dist(minValue, maxValue);
        vector<T> vec(size);
        for (auto &elem : vec) {
            elem = dist(gen);  // 为每个元素生成随机数
        }
        return std::move(vec);
    } else if constexpr (std::is_floating_point<T>::value) {
        // 对于浮点类型，使用均匀浮点分布
        std::uniform_real_distribution<T> dist(minValue, maxValue);
        vector<T> vec(size);
        for (auto &elem : vec) {
            elem = dist(gen);  // 为每个元素生成随机数
        }
        return std::move(vec);
    } else {
        // 如果 T 不是整型或浮点型，可以根据需求报错或返回空向量
        LUISA_ASSERT(false, "Unsupported type for random vector generation");
        return vector<T>();
    }
}

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

    // read_csv("data/TRD_Dalyr0.csv", table);

    uint seed = 0;
    uint size = std::stoi(argv[2]);
    auto id_vec = createRandomVector<int32_t>(size, 0, std::stoi(argv[3]), seed);
    table.create_column("id", TypeId::INT32);
    table.append_column("id", id_vec);

    auto opnprc_vec = createRandomVector<float>(size, 0, 1, seed + 20);
    table.create_column("opnprc", TypeId::FLOAT32);
    table.append_column("opnprc", opnprc_vec);

    auto clsprc_vec = createRandomVector<float>(size, -100, 100, seed + 1);
    table.create_column("clsprc", TypeId::FLOAT32);
    table.append_column("clsprc", clsprc_vec);

    table.print_table();

    table.where("id", FilterOp::LESS_EQUAL, std::stoi(argv[4]));

    table.print_table();
    
    // int64_t th = std::stoi(argv[2]);
    // // long long th2 = std::stoi(argv[3]);
    // // for (int i = 0; i < 100; ++i) {
    // //     clock.tic();
    // //     table.query().where("id", FilterOp::LESS, &th);
    // //     LUISA_INFO("Time: {} ms", clock.toc());
    // // }
    // table.where("id", FilterOp::LESS, &th);

    // table.print_table();

    // print_buffer(stream, res.view().as<int>());
    std::cout << "End.\n";
}