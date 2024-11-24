#ifndef _CPPNN_UTILITY_HPP
#define _CPPNN_UTILITY_HPP

#include <bit>
#include <concepts>
#include <iostream>
#include <random>
namespace cppnn
{
using namespace std;
// Support for random number generation
static random_device::result_type seed = (random_device())();
static mt19937 gen(seed);
inline double uniform_random(double min, double max)
{
    uniform_real_distribution<double> d(min, max);
    return d(gen);
}
inline double normal_random(double mean, double stddev)
{
    normal_distribution d(mean, stddev);
    return d(gen);
}
// Set the number to zero
inline void zero(double &a)
{
    a = 0;
}
// Although `std::ceil` and `std::floor` are constexpr since C++23, clang don't support it
constexpr size_t constexpr_ceil(size_t dividend, size_t divisor)
{
    return (dividend + divisor - 1) / divisor;
}
constexpr size_t constexpr_floor(size_t dividend, size_t divisor)
{
    return dividend / divisor;
}
// Helper function for calculating output feature map size
static constexpr size_t feature_map_size(size_t image_size, size_t kernel_size, size_t padding, size_t stride)
{
    return (image_size - kernel_size + padding + stride) / stride;
}
// Support for reading and writing data
template <integral T, endian endian>
T read(istream &in)
{
    T val;
    in.read(reinterpret_cast<char *>(&val), sizeof(T));
    return endian::native == endian ? val : byteswap(val);
}
template <floating_point T, endian endian>
T read(istream &in)
{
    if constexpr (sizeof(T) == 4)
        return bit_cast<T>(read<int32_t, endian>(in));
    else if constexpr (sizeof(T) == 8)
        return bit_cast<T>(read<int64_t, endian>(in));
    else
        static_assert(false, "Unsupported floating point type");
}
template <integral T, endian endian>
void write(ostream &out, T val)
{
    if constexpr (endian::native != endian)
        val = byteswap(val);
    out.write(reinterpret_cast<char *>(&val), sizeof(T));
}
template <floating_point T, endian endian>
void write(ostream &out, T val)
{
    if constexpr (sizeof(T) == 4)
        return write<int32_t, endian>(out, bit_cast<int32_t>(val));
    else if constexpr (sizeof(T) == 8)
        return write<int64_t, endian>(out, bit_cast<int64_t>(val));
    else
        static_assert(false, "Unsupported floating point type");
}
} // namespace cppnn

#endif // _CPPNN_UTILITY_HPP