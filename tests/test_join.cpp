
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

    auto set_bit = device.compile<1>([](Var<Bitmap> bitmap) {
        auto x = dispatch_x();
        bitmap->set(x);
    });

    Table table(device, stream);
    // table.create_column("id", TypeId::INT32);
    // table.append_column("id", vector<int>{1, 1, 2, 3});
    // table.create_column("opnprc", TypeId::FLOAT32);
    // table.append_column("opnprc", vector<float>{1561.156486, 1.5, 2.5, 3.5});

    uint seed = 0;
    uint size = std::stoi(argv[2]);
    auto id1_vec = createRandomVector<int32_t>(size, 0, std::stoi(argv[3]), seed);
    table.create_column("id", TypeId::INT32);
    table.append_column("id", id1_vec);

    auto opnprc_vec = createRandomVector<float>(size, -100, 100, seed + 20);
    table.create_column("opnprc", TypeId::FLOAT32);
    table.append_column("opnprc", opnprc_vec);

    table._columns["opnprc"]._null_mask.init_zero(device, stream, table._columns["opnprc"].size(), ShaderCollector<uint>::get_instance(device)->set_shader);
    stream << set_bit(table._columns["opnprc"]._null_mask).dispatch(2) << synchronize();
    print_buffer(stream, table._columns["opnprc"]._null_mask._data.view());

    // table.print_table();


    Table table2(device, stream);
    // table2.create_column("id2", TypeId::INT32);
    // table2.append_column("id2", vector<int>{2, 2, 3, 4, 4});
    // table2.create_column("clsprc", TypeId::FLOAT32);
    // table2.append_column("clsprc", vector<float>{1.8, 2.8, 3.8, 4.8, 5.8});

    seed = 10;
    size = std::stoi(argv[4]);
    auto id_vec = createRandomVector<int32_t>(size, 0, std::stoi(argv[5]), seed);
    table2.create_column("id2", TypeId::INT32);
    table2.append_column("id2", id_vec);
    auto clsprc_vec = createRandomVector<float>(size, 0, 10, seed + 20);
    table2.create_column("clsprc", TypeId::FLOAT32);
    table2.append_column("clsprc", clsprc_vec);

    table2._columns["clsprc"]._null_mask.init_zero(device, stream, table2._columns["clsprc"].size(), ShaderCollector<uint>::get_instance(device)->set_shader);
    stream << set_bit(table2._columns["clsprc"]._null_mask).dispatch(1) << synchronize();
    print_buffer(stream, table2._columns["clsprc"]._null_mask._data.view());

    // table2.print_table();

    clock.tic();
    table.join(table2, "id", "id2", JoinType::LEFT);
    LUISA_INFO("join in {} ms", clock.toc());
    // table.sort(argv[6], SortOrder::Ascending);
    table.print_table();
    // table.where("opnprc", FilterOp::GREATER_EQUAL, 7.0f);
    auto agg_op_map = unordered_map<string, vector<AggeragateOp>>();
    // agg_op_map.insert({"opnprc", {AggeragateOp::SUM, AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::COUNT, AggeragateOp::MEAN}});
    agg_op_map.insert({"clsprc", {AggeragateOp::SUM, AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::COUNT, AggeragateOp::MEAN}});
    agg_op_map.insert({"opnprc", {AggeragateOp::SUM, AggeragateOp::MAX, AggeragateOp::MIN, AggeragateOp::COUNT, AggeragateOp::MEAN}});
    // agg_op_map.insert({"opnprc", {AggeragateOp::SUM, AggeragateOp::MAX, AggeragateOp::MIN}});
    clock.tic();
    table.group_by(argv[6], agg_op_map);
    LUISA_INFO("group by in {} ms", clock.toc());
    table.print_table();

    std::cout << "End.\n";
}