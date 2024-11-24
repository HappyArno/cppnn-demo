#ifndef _CPPNN_LOSS_HPP
#define _CPPNN_LOSS_HPP

#include "data_structure.hpp"
#include "layer.hpp"
namespace cppnn
{
using namespace std;
// Mean squared error
struct MSELoss
{
    static double execute(double output, double target)
    {
        double tmp = output - target;
        return tmp * tmp;
    }
    template <size_t N>
    static double execute(const Vec<N> &output, const Vec<N> &target)
    {
        double sum = 0;
        for (size_t i = 0; i < N; i++)
            sum += execute(output[i], target[i]);
        return sum;
    }
    template <size_t H, size_t W>
    static double execute(const Image<H, W> &output, const Image<H, W> &target)
    {
        double sum = 0;
        for (size_t i = 0; i < H; i++)
            for (size_t j = 0; j < W; j++)
                sum += execute(output[i][j], target[i][j]);
        return sum;
    }
    static double derivative(double output, double target)
    {
        return 2 * (output - target);
    }
    template <size_t N>
    static Vec<N> gradient(const Vec<N> &output, const Vec<N> &target)
    {
        Vec<N> ret;
        for (size_t i = 0; i < N; i++)
            ret[i] = derivative(output[i], target[i]);
        return ret;
    }
    template <size_t H, size_t W>
    static Image<H, W> gradient(const Image<H, W> &output, const Image<H, W> &target)
    {
        Image<H, W> ret;
        for (size_t i = 0; i < H; i++)
            for (size_t j = 0; j < W; j++)
                ret[i][j] = derivative(output[i][j], target[i][j]);
        return ret;
    }
};
// Cross entropy loss
struct SoftmaxCategoricalCrossEntropyLoss
{
    template <size_t N>
    static double execute(Vec<N> logits, Vec<N> target)
    {
        Vec<N> log_softmax = LogSoftmax::forward(logits);
        double loss = 0;
        for (size_t i = 0; i < N; i++)
            loss += target[i] * log_softmax[i];
        return -loss;
    }
    template <size_t N>
    static Vec<N> gradient(Vec<N> logits, Vec<N> target)
    {
        return LogSoftmax::bp(logits, -target);
    }
};
} // namespace cppnn

#endif // _CPPNN_LOSS_HPP