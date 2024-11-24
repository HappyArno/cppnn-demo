#include "mnist_model.hpp"
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <variant>
#include <vector>
using namespace std;
const char *help = R"(Usage: ./mnist <command> [options] [operands]
Commands:
    help: Display help information
    train: Train a model
        Options:
        --model_type <string>: Specify the model type (MLP/CNN)
        --model_path <string>: Specify the model path
        --image_path <string>: Specify the training set image file path
        --label_path <string>: Specify the training set label file path
        --epoch_size <integer>: Specify the epoch size
        --batch_size <integer>: Specify the batch size
        --learning_rate <double>: Specify the learning rate
        --weight_decay_rate <double>: Specify the weight decay rate
        --seed <integer>: Specify the seed for generating random numbers
    test: Test a model
        Options:
        --model_type <string>: Specify the model type (MLP/CNN)
        --model_path <string>: Specify the model path
        --image_path <string>: Specify the test set image file path
        --label_path <string>: Specify the test set label file path
    run:
        Operands: The path of the file to be recognized
        Options:
        --model_type <string>: Specify the model type (MLP/CNN)
        --model_path <string>: Specify the model path
)";
// Parse arguments
vector<string> get_arg(int argc, char *argv[], map<string, variant<string, size_t, double>> &config)
{
    vector<string> operands;
    for (size_t i = 0; i < argc; i++)
    {
        if (argv[i][0] == '-' && argv[i][1] == '-')
        {
            if (i + 1 < argc)
            {
                auto &value = config.at(string(argv[i] + 2));
                switch (value.index())
                { // clang-format off
                case 0: value = argv[i + 1]; break;
                case 1: value = stoull(argv[i + 1]); break;
                case 2: value = stod(argv[i + 1]); break;
                } // clang-format on
                i++;
            }
            else
                throw runtime_error("The option needs an argument");
        }
        else
            operands.push_back(argv[i]);
    }
    return operands;
}
template <typename ModelType>
void train(const map<string, variant<string, size_t, double>> &config, const vector<string> &operands)
{
    cppnn::seed = get<size_t>(config.at("seed"));
    cppnn::gen = mt19937(cppnn::seed);
    ModelType model;
    model.xavier_init();
    cppnn::mnist::Dataset train_dataset(
        ifstream(get<string>(config.at("image_path")), ios::binary),
        ifstream(get<string>(config.at("label_path")), ios::binary)
    );
    model.train(
        train_dataset,
        get<size_t>(config.at("epoch_size")),
        get<size_t>(config.at("batch_size")),
        get<double>(config.at("learning_rate")),
        get<double>(config.at("weight_decay_rate"))
    );
    model.save(get<string>(config.at("model_path")));
}
template <typename ModelType>
void test(const map<string, variant<string, size_t, double>> &config, const vector<string> &operands)
{
    ModelType model;
    model.load(get<string>(config.at("model_path")));
    cppnn::mnist::Dataset test_dataset(
        ifstream(get<string>(config.at("image_path")), ios::binary),
        ifstream(get<string>(config.at("label_path")), ios::binary)
    );
    model.test(test_dataset);
}
template <typename ModelType>
void run(const map<string, variant<string, size_t, double>> &config, const vector<string> &operands)
{
    ModelType model;
    model.load(get<string>(config.at("model_path")));
    for (auto &img : operands)
    {
        cout << img << ": "
             << cppnn::mnist::parse_pgm(ifstream(img, ios::binary)).apply(model).max()
             << '\n';
    }
}
void model_operation(int argc, char *argv[], string type, map<string, variant<string, size_t, double>> config)
{
    vector<string> operands = get_arg(argc - 2, argv + 2, config);
    // print the configuration
    for (const auto &[name, value] : config)
    {
        cout << name << ": ";
        visit([](auto &&value) { cout << value; }, value);
        cout << '\n';
    }
    cout << '\n';
    cout << fixed << setprecision(3);
    // execute corresponding functions according to models and operations
    map<string, map<string, function<void(const map<string, variant<string, size_t, double>> &config, const vector<string> &operands)>>> models{
        {"MLP", {{"train", train<cppnn::mnist::MLP>}, {"test", test<cppnn::mnist::MLP>}, {"run", run<cppnn::mnist::MLP>}}},
        {"CNN", {{"train", train<cppnn::mnist::CNN>}, {"test", test<cppnn::mnist::CNN>}, {"run", run<cppnn::mnist::CNN>}}}
    };
    (models.at(get<string>(config.at("model_type"))).at(type))(config, operands);
}
int main(int argc, char *argv[])
{
    // store all commands
    map<string, function<void()>> commands{
        {"help", [] { cout << help; }},
        {"train", [argc, argv] {
             model_operation(
                 argc, argv, "train",
                 {{"model_type", "MLP"},
                  {"model_path", "model.bin"},
                  {"image_path", "train-images-idx3-ubyte"},
                  {"label_path", "train-labels-idx1-ubyte"},
                  {"epoch_size", 1u},
                  {"batch_size", 100u},
                  {"learning_rate", 0.5},
                  {"weight_decay_rate", 0.001},
                  {"seed", cppnn::seed}}
             );
         }},
        {"test", [argc, argv] {
             model_operation(
                 argc, argv, "test",
                 {{"model_type", "MLP"},
                  {"model_path", "model.bin"},
                  {"image_path", "t10k-images-idx3-ubyte"},
                  {"label_path", "t10k-labels-idx1-ubyte"}}
             );
         }},
        {"run", [argc, argv] {
             model_operation(
                 argc, argv, "run",
                 {{"model_type", "MLP"},
                  {"model_path", "model.bin"}}
             );
         }}
    };
    // execute the command
    if (argc < 2)
        throw runtime_error("Please input command");
    (commands[argv[1]])();
    return 0;
}