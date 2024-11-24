#ifndef _CPPNN_LAYER_HPP
#define _CPPNN_LAYER_HPP

#include "data_structure.hpp"
namespace cppnn
{
using namespace std;
// Affine linear transformation
template <size_t in, size_t out>
struct Linear
{
    Matrix<out, in> weights;
    Vec<out> bias;
    template <typename Func>
    void for_each(Func &&f)
    {
        weights.for_each(f);
        bias.for_each(f);
    }
    void zero()
    {
        for_each(cppnn::zero);
    }
    Linear<in, out> &operator+=(const Linear<in, out> &rhs)
    {
        weights += rhs.weights;
        bias += rhs.bias;
        return *this;
    }
    Linear<in, out> &operator*=(const double rhs)
    {
        for_each([rhs](double &a) { a *= rhs; });
        return *this;
    }
    Vec<out> forward(const Vec<in> &v) const
    {
        Vec<out> ret = weights * v;
        ret += bias;
        return ret;
    }
    void display() const
    {
        cout << "weights:" << '\n';
        weights.display();
        cout << "bias:" << '\n';
        bias.display();
    }
    static Linear<in, out> gradient(const Vec<in> &prev_layer_activation, const Vec<out> &next_layer_linear_gradient)
    {
        Linear<in, out> gradient;
        for (size_t i = 0; i < out; i++)
        {
            for (size_t j = 0; j < in; j++)
                gradient.weights[i][j] = prev_layer_activation[j] * next_layer_linear_gradient[i];
            gradient.bias[i] = next_layer_linear_gradient[i];
        }
        return gradient;
    }
    Vec<in> bp(const Vec<out> &next_layer_linear_gradient) const
    {
        Vec<in> gradient;
        for (size_t i = 0; i < in; i++)
        {
            gradient[i] = 0;
            for (size_t j = 0; j < out; j++)
                gradient[i] += weights[j][i] * next_layer_linear_gradient[j];
        }
        return gradient;
    }
};
// Two-dimensional convolution, or actually two-dimensional cross-correlation
template <size_t channel_input, size_t channel_output, size_t image_h, size_t image_w, size_t kernel_h, size_t kernel_w, size_t padding_h = kernel_h - 1, size_t padding_w = kernel_w - 1, size_t stride_h = 1, size_t stride_w = 1>
struct Conv2d
{
    static constexpr size_t feature_map_h = feature_map_size(image_h, kernel_h, padding_h, stride_h);
    static constexpr size_t feature_map_w = feature_map_size(image_w, kernel_w, padding_w, stride_w);
    static const ssize_t padding_top = constexpr_ceil(padding_h, 2), padding_bottom = constexpr_floor(padding_h, 2);
    static const ssize_t padding_left = constexpr_ceil(padding_w, 2), padding_right = constexpr_floor(padding_w, 2);
    using Self = Conv2d<channel_input, channel_output, image_h, image_w, kernel_h, kernel_w, padding_h, padding_w, stride_h, stride_w>;
    using InputType = Images<channel_input, image_h, image_w>;
    using ReturnType = Images<channel_output, feature_map_h, feature_map_w>;
    Image<kernel_h, kernel_w> kernels[channel_input][channel_output];
    Images<channel_output, feature_map_h, feature_map_w> bias;
    template <typename Func>
    void for_each(Func &&f)
    {
        for (size_t i = 0; i < channel_input; i++)
            for (size_t j = 0; j < channel_output; j++)
                kernels[i][j].for_each(f);
        bias.for_each(f);
    }
    void zero()
    {
        for_each(cppnn::zero);
    }
    Self &operator+=(const Self &rhs)
    {
        for (size_t i = 0; i < channel_input; i++)
            for (size_t j = 0; j < channel_output; j++)
                kernels[i][j] += rhs.kernels[i][j];
        bias += rhs.bias;
        return *this;
    }
    Self &operator*=(const double rhs)
    {
        for_each([rhs](double &a) { a *= rhs; });
        return *this;
    }
    static Image<feature_map_h, feature_map_w> kernel_forward(const Image<image_h, image_w> &img, const Image<kernel_h, kernel_w> &kernel)
    {
        Image<feature_map_h, feature_map_w> feature_map;
        for (size_t i = 0; i < feature_map_h; i++)
            for (size_t j = 0; j < feature_map_w; j++)
            {
                feature_map[i][j] = 0;
                for (ssize_t offset_i = 0; offset_i < kernel_h; offset_i++)
                    for (ssize_t offset_j = 0; offset_j < kernel_w; offset_j++)
                        feature_map[i][j] += kernel[offset_i][offset_j] * img.get(i * stride_h + offset_i - padding_top, j * stride_w + offset_j - padding_left);
            }
        return feature_map;
    }
    ReturnType forward(const InputType &img) const
    {
        ReturnType feature_map;
        for (size_t i = 0; i < channel_output; i++)
        {
            feature_map[i].zero();
            for (size_t j = 0; j < channel_input; j++)
                feature_map[i] += kernel_forward(img[j], kernels[j][i]);
        }
        feature_map += bias;
        return feature_map;
    }
    static Image<kernel_h, kernel_w> kernel_gradient(const Image<image_h, image_w> &prev_layer_activation, const Image<feature_map_h, feature_map_w> &next_layer_linear_gradient)
    {
        Image<kernel_h, kernel_w> gradient;
        gradient.zero();
        for (size_t i = 0; i < feature_map_h; i++)
            for (size_t j = 0; j < feature_map_w; j++)
            {
                for (ssize_t offset_i = 0; offset_i < kernel_h; offset_i++)
                    for (ssize_t offset_j = 0; offset_j < kernel_w; offset_j++)
                        gradient[offset_i][offset_j] += prev_layer_activation.get(i * stride_h + offset_i - padding_top, j * stride_w + offset_j - padding_left) * next_layer_linear_gradient[i][j];
            }
        return gradient;
    }
    static Self gradient(const InputType &prev_layer_activation, const ReturnType &next_layer_linear_gradient)
    {
        Self gradient;
        for (size_t i = 0; i < channel_input; i++)
            for (size_t j = 0; j < channel_output; j++)
                gradient.kernels[i][j] = kernel_gradient(prev_layer_activation[i], next_layer_linear_gradient[j]);
        gradient.bias = next_layer_linear_gradient;
        return gradient;
    }
    static Image<image_h, image_w> kernel_bp(const Image<kernel_h, kernel_w> &kernel, const Image<feature_map_h, feature_map_w> &next_layer_linear_gradient)
    {
        Image<image_h, image_w> gradient;
        gradient.zero();
        for (size_t i = 0; i < feature_map_h; i++)
            for (size_t j = 0; j < feature_map_w; j++)
                for (ssize_t offset_i = 0; offset_i < kernel_h; offset_i++)
                    for (ssize_t offset_j = 0; offset_j < kernel_w; offset_j++)
                    {
                        ssize_t index_h = i * stride_h + offset_i - padding_top;
                        ssize_t index_w = j * stride_w + offset_j - padding_left;
                        if (0 <= index_h && index_h < image_h && 0 <= index_w && index_w < image_w)
                            gradient[index_h][index_w] += kernel[offset_i][offset_j] * next_layer_linear_gradient[i][j];
                    }
        return gradient;
    }
    InputType bp(const ReturnType &next_layer_linear_gradient) const
    {
        InputType gradient;
        for (size_t i = 0; i < channel_input; i++)
        {
            gradient[i].zero();
            for (size_t j = 0; j < channel_output; j++)
                gradient[i] += kernel_bp(kernels[i][j], next_layer_linear_gradient[j]);
        }
        return gradient;
    }
};
// Two-dimensional max pooling
template <size_t image_h, size_t image_w, size_t kernel_h, size_t kernel_w, size_t stride_h = kernel_h, size_t stride_w = kernel_w, size_t padding_h = 0, size_t padding_w = 0>
struct MaxPool2d
{
    static constexpr size_t feature_map_h = feature_map_size(image_h, kernel_h, padding_h, stride_h);
    static constexpr size_t feature_map_w = feature_map_size(image_w, kernel_w, padding_w, stride_w);
    static constexpr ssize_t padding_top = constexpr_ceil(padding_h, 2), padding_bottom = constexpr_floor(padding_h, 2);
    static constexpr ssize_t padding_left = constexpr_ceil(padding_w, 2), padding_right = constexpr_floor(padding_w, 2);
    static auto forward(const Image<image_h, image_w> &img)
    {
        Image<feature_map_h, feature_map_w> feature_map;
        for (size_t i = 0; i < feature_map_h; i++)
            for (size_t j = 0; j < feature_map_w; j++)
            {
                feature_map[i][j] = -numeric_limits<double>::infinity();
                for (ssize_t offset_i = 0; offset_i < kernel_h; offset_i++)
                    for (ssize_t offset_j = 0; offset_j < kernel_w; offset_j++)
                        feature_map[i][j] = max(feature_map[i][j], img.get(i * stride_h + offset_i - padding_top, j * stride_w + offset_j - padding_left, -numeric_limits<double>::infinity()));
            }
        return feature_map;
    }
    template <size_t N>
    static auto forward(const Images<N, image_h, image_w> &img)
    {
        Images<N, feature_map_h, feature_map_w> feature_map;
        for (size_t i = 0; i < N; i++)
            feature_map[i] = forward(img[i]);
        return feature_map;
    }
    static Image<image_h, image_w> bp(const Image<image_h, image_w> &img, const Image<feature_map_h, feature_map_w> &next_layer_linear_gradient)
    {
        Image<image_h, image_w> gradient;
        gradient.zero();
        for (size_t i = 0; i < feature_map_h; i++)
            for (size_t j = 0; j < feature_map_w; j++)
            {
                double max_value = -numeric_limits<double>::infinity();
                size_t max_index_i, max_index_j;
                for (ssize_t offset_i = 0; offset_i < kernel_h; offset_i++)
                    for (ssize_t offset_j = 0; offset_j < kernel_w; offset_j++)
                    {
                        ssize_t index_i = i * stride_h + offset_i - padding_top, index_j = j * stride_w + offset_j - padding_left;
                        double value = img.get(index_i, index_j, -numeric_limits<double>::infinity());
                        if (value > max_value)
                        {
                            max_value = value;
                            max_index_i = index_i, max_index_j = index_j;
                        }
                    }
                if (0 <= max_index_i && max_index_i < image_h && 0 <= max_index_j && max_index_j < image_w)
                    gradient[max_index_i][max_index_j] = next_layer_linear_gradient[i][j];
            }
        return gradient;
    }
    template <size_t N>
    static Images<N, image_h, image_w> bp(const Images<N, image_h, image_w> &img, const Images<N, feature_map_h, feature_map_w> &next_layer_linear_gradient)
    {
        Images<N, image_h, image_w> feature_map;
        for (size_t i = 0; i < N; i++)
            feature_map[i] = bp(img[i], next_layer_linear_gradient[i]);
        return feature_map;
    }
};
// Dropout layer
template <size_t N, double probability> // probability of an element to be zeroed
struct Dropout
{
    static_assert(0 <= probability && probability <= 1, "The range of probability must be between 0 and 1");
    bool is_dropout[N];
    Dropout()
    {
        for (size_t i = 0; i < N; i++)
            is_dropout[i] = (uniform_random(0, 1) <= probability);
    }
    Vec<N> forward(Vec<N> v) const
    {
        for (size_t i = 0; i < N; i++)
            if (is_dropout[i])
                v[i] = 0;
            else
                v[i] /= 1 - probability;
        return v;
    }
    Vec<N> bp(const Vec<N> &next_layer_gradient) const
    {
        Vec<N> gradient;
        for (size_t i = 0; i < N; i++)
            if (is_dropout[i])
                gradient[i] = 0;
            else
                gradient[i] = next_layer_gradient[i] / (1 - probability);
        return gradient;
    }
};
template <size_t H, size_t W, double probability> // probability of an element to be zeroed
struct Dropout2d
{
    static_assert(0 <= probability && probability <= 1, "The range of probability must be between 0 and 1");
    bool is_dropout[H][W];
    Dropout2d()
    {
        for (size_t i = 0; i < H; i++)
            for (size_t j = 0; j < W; j++)
                is_dropout[i][j] = (uniform_random(0, 1) <= probability);
    }
    Image<H, W> forward(Image<H, W> v)
    {
        for (size_t i = 0; i < H; i++)
            for (size_t j = 0; j < W; j++)
                if (is_dropout[i][j])
                    v[i][j] = 0;
                else
                    v[i][j] /= 1 - probability;
        return v;
    }
    Image<H, W> bp(const Image<H, W> &next_layer_gradient)
    {
        Image<H, W> gradient;
        for (size_t i = 0; i < H; i++)
            for (size_t j = 0; j < W; j++)
                if (is_dropout[i])
                    gradient[i] = 0;
                else
                    gradient[i] = next_layer_gradient[i] / (1 - probability);
        return gradient;
    }
};
// Activation functions
struct sigmoid
{
    static double forward(double x)
    {
        return 1 / (1 + pow(numbers::e, -x));
    }
    template <size_t N>
    static Vec<N> forward(Vec<N> v)
    {
        for (auto &i : v)
            i = forward(i);
        return v;
    }
    static double derivative(double x)
    {
        double e_x = exp(x);
        double tmp = 1 + e_x;
        return e_x / (tmp * tmp);
    }
    template <size_t N>
    static Vec<N> derivative(Vec<N> v)
    {
        for (auto &i : v)
            i = derivative(i);
        return v;
    }
    template <size_t N>
    static Vec<N> bp(const Vec<N> &layer_linear, const Vec<N> &layer_gradient)
    {
        Vec<N> gradient;
        for (size_t i = 0; i < N; i++)
            gradient[i] = derivative(layer_linear[i]) * layer_gradient[i];
        return gradient;
    }
};
template <double negative_slope = 0.01>
struct LeakyReLU
{
    static double forward(double x)
    {
        return x > 0 ? x : negative_slope * x;
    }
    template <size_t N>
    static Vec<N> forward(Vec<N> v)
    {
        for (auto &i : v)
            i = forward(i);
        return v;
    }
    template <size_t H, size_t W>
    static Image<H, W> forward(Image<H, W> img)
    {
        for (auto &i : img)
            for (auto &j : i)
                j = forward(j);
        return img;
    }
    template <size_t N, size_t H, size_t W>
    static Images<N, H, W> forward(Images<N, H, W> img)
    {
        for (auto &i : img)
            i = forward(i);
        return img;
    }
    static double derivative(double x)
    {
        return x > 0 ? 1 : negative_slope;
    }
    template <size_t N>
    static Vec<N> derivative(Vec<N> v)
    {
        for (auto &i : v)
            i = derivative(i);
        return v;
    }
    template <size_t N>
    static Vec<N> bp(const Vec<N> &layer_linear, const Vec<N> &layer_gradient)
    {
        Vec<N> gradient;
        for (size_t i = 0; i < N; i++)
            gradient[i] = derivative(layer_linear[i]) * layer_gradient[i];
        return gradient;
    }
    template <size_t H, size_t W>
    static Image<H, W> bp(const Image<H, W> &layer_linear, const Image<H, W> &layer_gradient)
    {
        Image<H, W> gradient;
        for (size_t i = 0; i < H; i++)
            for (size_t j = 0; j < W; j++)
                gradient[i][j] = derivative(layer_linear[i][j]) * layer_gradient[i][j];
        return gradient;
    }
    template <size_t N, size_t H, size_t W>
    static Images<N, H, W> bp(const Images<N, H, W> &layer_linear, const Images<N, H, W> &layer_gradient)
    {
        Images<N, H, W> gradient;
        for (size_t i = 0; i < N; i++)
            gradient[i] = bp(layer_linear[i], layer_gradient[i]);
        return gradient;
    }
};
using ReLU = LeakyReLU<0.>;
struct softmax
{
    template <size_t N>
    static Vec<N> forward(Vec<N> v)
    {
        Vec<N> z_exp, softmax;
        double sum_z_exp = 0, max = v[v.max()];
        for (size_t i = 0; i < N; i++)
        {
            z_exp[i] = exp(v[i] - max); // subtract the maximum value to prevent overflow
            sum_z_exp += z_exp[i];
        }
        for (size_t i = 0; i < N; i++)
            softmax[i] = z_exp[i] / sum_z_exp;
        return softmax;
    }
    // the partial derivative of softmax_j with respect to x_i
    template <size_t N>
    static double derivative(Vec<N> softmax, size_t i, size_t j)
    {
        if (i == j)
            return softmax[j] * (1 - softmax[i]);
        else
            return -softmax[i] * softmax[j];
    }
    template <size_t N>
    static Vec<N> bp(const Vec<N> &layer_linear, const Vec<N> &layer_activation, const Vec<N> &layer_gradient)
    {
        Vec<N> gradient;
        for (size_t i = 0; i < N; i++)
        {
            gradient[i] = 0;
            for (size_t j = 0; j < N; j++)
                gradient[i] += derivative(layer_activation, i, j) * layer_gradient[j];
        }
        return gradient;
    }
};
struct LogSoftmax
{
    template <size_t N>
    static Vec<N> forward(Vec<N> v)
    {
        double sum = 0, max = v[v.max()];
        for (size_t i = 0; i < N; i++)
            sum += exp(v[i] - max); // subtract the maximum value to prevent overflow
        sum = log(sum);
        Vec<N> logSoftmax;
        for (size_t i = 0; i < N; i++)
            logSoftmax[i] = v[i] - max - sum;
        return logSoftmax;
    }
    // the partial derivative of LogSoftmax_j with respect to x_i
    template <size_t N>
    static double derivative(Vec<N> softmax, size_t i, size_t j)
    {
        if (i == j)
            return 1 - softmax[i];
        else
            return -softmax[i];
    }
    template <size_t N>
    static Vec<N> bp(const Vec<N> &layer_linear, const Vec<N> &layer_gradient)
    {
        Vec<N> softmax = softmax::forward(layer_linear);
        Vec<N> gradient;
        for (size_t i = 0; i < N; i++)
        {
            gradient[i] = 0;
            for (size_t j = 0; j < N; j++)
                gradient[i] += derivative(softmax, i, j) * layer_gradient[j];
        }
        return gradient;
    }
};
} // namespace cppnn

#endif // _CPPNN_LAYER_HPP