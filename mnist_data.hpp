#ifndef _CPPNN_MNIST_DATA_HPP
#define _CPPNN_MNIST_DATA_HPP

#include "data_structure.hpp"
#include <cstdint>
#include <istream>
namespace cppnn::mnist
{
using namespace std;
// MNIST data
struct Data
{
    Image<28, 28> image;
    uint8_t label;
    Data(istream &images, istream &labels)
    {
        for (auto &i : image)
            for (auto &j : i)
                j = static_cast<double>(read<uint8_t, endian::big>(images)) / 255;
        label = read<uint8_t, endian::big>(labels);
        if (!(0 <= label && label <= 9))
            throw runtime_error("bad label");
    }
};
// MNIST dataset
struct Dataset : vector<Data>
{
    static constexpr int32_t MNIST_IMAGE_MAGIC_NUMBER = 2051;
    static constexpr int32_t MNIST_LABEL_MAGIC_NUMBER = 2049;
    Dataset(istream &images, istream &labels)
    {
        images.exceptions(istream::eofbit | istream::failbit | istream::badbit);
        labels.exceptions(istream::eofbit | istream::failbit | istream::badbit);
        if (read<int32_t, endian::big>(images) != MNIST_IMAGE_MAGIC_NUMBER)
            throw runtime_error("image magic number incorrect");
        if (read<int32_t, endian::big>(labels) != MNIST_LABEL_MAGIC_NUMBER)
            throw runtime_error("label magic number incorrect");
        uint32_t num = read<uint32_t, endian::big>(images);
        if (num != read<uint32_t, endian::big>(labels))
            throw runtime_error("the number of items inconsistent");
        reserve(num);
        uint32_t row = read<uint32_t, endian::big>(images);
        uint32_t column = read<uint32_t, endian::big>(images);
        if (!(row == 28 && column == 28))
            throw runtime_error("row or column incorrect");
        for (uint32_t i = 0; i < num; i++)
            push_back(Data(images, labels));
    }
    Dataset(istream &&images, istream &&labels) : Dataset(images, labels) {}
    void disorder()
    {
        shuffle(begin(), end(), gen);
    }
};
// Parse portable graymap format (PGM)
inline Image<28, 28> parse_pgm(istream &in)
{
    in.exceptions(istream::eofbit | istream::failbit | istream::badbit);
    string header = "P5\n28 28\n255\n";
    string obtained(header.size(), '\0');
    in.read(obtained.data(), obtained.size());
    if (obtained != header)
        throw runtime_error("unexpected header");
    Image<28, 28> image;
    for (auto &i : image)
        for (auto &j : i)
            j = static_cast<double>(255 - read<uint8_t, endian::big>(in)) / 255;
    return image;
}
inline Image<28, 28> parse_pgm(istream &&in)
{
    return parse_pgm(in);
}
} // namespace cppnn::mnist

#endif // _CPPNN_MNIST_DATA_HPP