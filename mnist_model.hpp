#ifndef _CPPNN_MNIST_MODEL_HPP
#define _CPPNN_MNIST_MODEL_HPP

#include "data_structure.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "mnist_data.hpp"
#include "utility.hpp"
#include <cmath>
#include <fstream>
namespace cppnn::mnist
{
using namespace std;
// Helper template for creating models
template <typename Derived> // CRTP
struct Model
{
    virtual void zero()
    {
        static_cast<Derived *>(this)->for_each(cppnn::zero);
    }
    // Xavier Initialization
    virtual void xavier_init()
    {
        static_cast<Derived *>(this)->for_each([](double &a) { a = normal_random(0, sqrt(2. / (28 * 28 + 10))); });
    }
    virtual Derived &operator+=(const Derived &rhs) = 0;
    virtual Derived &operator*=(const double rhs) = 0;
    virtual Vec<10> forward(const Image<28, 28> &img) const = 0;
    virtual double bp(const Data &data, Derived &gradient) const = 0;
    virtual void train(Dataset &train_dataset, size_t epoch_size, size_t batch_size, double learning_rate, double weight_decay_rate = 0, bool display = true)
    {
        Derived gradient;
        size_t size = train_dataset.size();
        for (size_t epoch = 0; epoch < epoch_size; epoch++)
        {
            if (display)
                cout << "[epoch " << epoch << "]" << endl;
            for (size_t batch = 0; batch < size / batch_size; batch++)
            {
                gradient.zero();
                double cost_avg = 0;
                for (size_t i = 0; i < batch_size; i++)
                {
                    size_t index = batch * batch_size + i;
                    cost_avg += bp(train_dataset[index], gradient);
                }
                cost_avg /= batch_size;
                if (display)
                    if ((batch + 1) % 100 == 0 || batch == size / batch_size - 1)
                        cout << "[batch " << batch << "] cost_avg: " << cost_avg << endl;
                (*this) *= 1 - learning_rate * weight_decay_rate; // weight decay
                gradient *= -learning_rate / batch_size;
                *this += gradient;
            }
            train_dataset.disorder();
        }
    }
    virtual void test(const Dataset &test_dataset)
    {
        size_t ac = 0, wa = 0;
        for (auto &i : test_dataset)
        {
            auto output = i.image.apply(*this);
            output.max() == i.label ? ac++ : wa++;
        }
        cout << "AC: " << ac << ", WA: " << wa << ", Accuracy: " << (double)ac / test_dataset.size() << endl;
    }
    virtual void save(ostream &out)
    {
        out.exceptions(istream::eofbit | istream::failbit | istream::badbit);
        static_cast<Derived *>(this)->for_each([&out](const double &a) { write<double, endian::little>(out, a); });
    }
    virtual void save(const string_view path)
    {
        ofstream out(path.data(), ios::binary);
        save(out);
    }
    virtual void load(istream &in)
    {
        in.exceptions(istream::eofbit | istream::failbit | istream::badbit);
        static_cast<Derived *>(this)->for_each([&in](double &a) { a = read<double, endian::little>(in); });
    }
    virtual void load(const string_view path)
    {
        ifstream in(path.data(), ios::binary);
        load(in);
    }
};
// Multilayer perceptron
struct MLP : Model<MLP>
{
    static constexpr size_t layer1_size = 16;
    static constexpr size_t layer2_size = 16;
    Linear<784, layer1_size> fc1;
    Linear<layer1_size, layer2_size> fc2;
    Linear<layer2_size, 10> fc3;
    template <typename Func>
    void for_each(Func &&f)
    {
        fc1.for_each(f);
        fc2.for_each(f);
        fc3.for_each(f);
    }
    MLP &operator+=(const MLP &rhs)
    {
        fc1 += rhs.fc1;
        fc2 += rhs.fc2;
        fc3 += rhs.fc3;
        return *this;
    }
    MLP &operator*=(const double rhs)
    {
        for_each([rhs](double &a) { a *= rhs; });
        return *this;
    }
    Vec<10> forward(const Image<28, 28> &img) const
    {
        return img.flatten().apply(fc1).apply<ReLU>().apply(fc2).apply<ReLU>().apply(fc3).apply<softmax>();
    }
    double bp(const Data &data, MLP &gradient) const
    {
        auto layer1_l = data.image.flatten().apply(fc1);
        auto layer1 = layer1_l.apply<ReLU>();
        auto layer2_l = layer1.apply(fc2);
        auto layer2 = layer2_l.apply<ReLU>();
        auto layer3_l = layer2.apply(fc3);
        // auto layer3 = layer3_l.apply<softmax>();
        Vec<10> target;
        target.fill(0);
        target[data.label] = 1;
        // auto layer3_gradient = MSELoss::gradient(layer3, target);
        // auto layer3_linear_gradient = softmax::bp(layer3_l, layer3,  layer3_gradient);
        auto layer3_linear_gradient = SoftmaxCategoricalCrossEntropyLoss::gradient(layer3_l, target);
        gradient.fc3 += fc3.gradient(layer2, layer3_linear_gradient);
        auto layer2_gradient = fc3.bp(layer3_linear_gradient);
        auto layer2_linear_gradient = ReLU::bp(layer2_l, layer2_gradient);
        gradient.fc2 += fc2.gradient(layer1, layer2_linear_gradient);
        auto layer1_gradient = fc2.bp(layer2_linear_gradient);
        auto layer1_linear_gradient = ReLU::bp(layer1_l, layer1_gradient);
        gradient.fc1 += fc1.gradient(data.image.flatten(), layer1_linear_gradient);
        return SoftmaxCategoricalCrossEntropyLoss::execute(layer3_l, target);
        // return MSELoss::execute(layer3, target);
    }
};
// Convolutional neural network
struct CNN : Model<CNN>
{
    Conv2d<1, 16, 28, 28, 3, 3> conv1;
    Conv2d<16, 32, 14, 14, 3, 3> conv2;
    Linear<7 * 7 * 32, 128> fc1;
    Linear<128, 10> fc2;
    template <typename Func>
    void for_each(Func &&f)
    {
        conv1.for_each(f);
        conv2.for_each(f);
        fc1.for_each(f);
        fc2.for_each(f);
    }
    CNN &operator+=(const CNN &rhs)
    {
        conv1 += rhs.conv1;
        conv2 += rhs.conv2;
        fc1 += rhs.fc1;
        fc2 += rhs.fc2;
        return *this;
    }
    CNN &operator*=(const double rhs)
    {
        for_each([rhs](double &a) { a *= rhs; });
        return *this;
    }
    Vec<10> forward(const Image<28, 28> &img) const
    {
        return Images<1, 28, 28>{img}.apply(conv1).apply<ReLU>().apply<MaxPool2d<28, 28, 2, 2>>().apply(conv2).apply<ReLU>().apply<MaxPool2d<14, 14, 2, 2>>().flatten().apply(fc1).apply<ReLU>().apply(fc2).apply<softmax>();
    }
    double bp(const Data &data, CNN &gradient) const
    {
        auto input = Images<1, 28, 28>{data.image};
        auto layer1_c = input.apply(conv1);
        auto layer1_a = layer1_c.apply<ReLU>();
        auto layer1 = MaxPool2d<28, 28, 2, 2>::forward(layer1_a);
        auto layer2_c = layer1.apply(conv2);
        auto layer2_a = layer2_c.apply<ReLU>();
        auto layer2 = MaxPool2d<14, 14, 2, 2>::forward(layer2_a);

        // auto layer3_l = layer2.flatten().apply(fc1);
        Dropout<32 * 7 * 7, 0.2> dropout1;
        auto layer2_d = layer2.flatten().apply(dropout1);
        auto layer3_l = layer2_d.apply(fc1);

        auto layer3 = layer3_l.apply<ReLU>();

        // auto layer4_l = layer3.apply(fc2);
        Dropout<128, 0.3> dropout2;
        auto layer3_d = layer3.apply(dropout2);
        auto layer4_l = layer3_d.apply(fc2);

        Vec<10> target;
        target.fill(0);
        target[data.label] = 1;
        auto layer4_linear_gradient = SoftmaxCategoricalCrossEntropyLoss::gradient(layer4_l, target);
        gradient.fc2 += fc2.gradient(layer3, layer4_linear_gradient);

        // auto layer3_gradient = fc2.bp(layer4_linear_gradient);
        auto layer3_dropout_gradient = fc2.bp(layer4_linear_gradient);
        auto layer3_gradient = dropout2.bp(layer3_dropout_gradient);

        auto layer3_linear_gradient = ReLU::bp(layer3_l, layer3_gradient);
        gradient.fc1 += fc1.gradient(layer2.flatten(), layer3_linear_gradient);

        // auto layer2_gradient = layer2.reshape(fc1.bp(layer3_linear_gradient));
        auto layer2_dropout_gradient = fc1.bp(layer3_linear_gradient);
        auto layer2_gradient = layer2.reshape(dropout1.bp(layer2_dropout_gradient));

        auto layer2_activation_gradient = MaxPool2d<14, 14, 2, 2>::bp(layer2_a, layer2_gradient);
        auto layer2_convolution_gradient = ReLU::bp(layer2_c, layer2_activation_gradient);
        gradient.conv2 += conv2.gradient(layer1, layer2_convolution_gradient);
        auto layer1_gradient = conv2.bp(layer2_convolution_gradient);
        auto layer1_activation_gradient = MaxPool2d<28, 28, 2, 2>::bp(layer1_a, layer1_gradient);
        auto layer1_convolution_gradient = ReLU::bp(layer1_c, layer1_activation_gradient);
        gradient.conv1 += conv1.gradient(input, layer1_convolution_gradient);
        return SoftmaxCategoricalCrossEntropyLoss::execute(layer4_l, target);
    }
};
} // namespace cppnn::mnist

#endif // _CPPNN_MNIST_MODEL_HPP