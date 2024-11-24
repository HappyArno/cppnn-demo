#ifndef _CPPNN_DATA_STRUCTURE_HPP
#define _CPPNN_DATA_STRUCTURE_HPP

#include "utility.hpp"
#include <array>
#include <iostream>
namespace cppnn
{
using namespace std;
// Vector
template <size_t N>
struct Vec : array<double, N>
{
    template <typename Func>
    void for_each(Func &&f)
    {
        for (auto &i : *this)
            f(i);
    }
    void zero()
    {
        for_each(cppnn::zero);
    }
    Vec<N> &operator+=(const Vec<N> &rhs)
    {
        for (size_t i = 0; i < N; i++)
            (*this)[i] += rhs[i];
        return *this;
    }
    Vec<N> &operator*=(const double rhs)
    {
        for_each([rhs](double &a) { a *= rhs; });
        return *this;
    }
    Vec<N> operator-() const
    {
        Vec<N> ret;
        for (size_t i = 0; i < N; i++)
            ret[i] = -(*this)[i];
        return ret;
    }
    template <typename T>
    auto apply(const T &val) const
    {
        return val.forward(*this);
    }
    template <typename Func>
    auto apply() const
    {
        return Func::forward(*this);
    }
    size_t max()
    {
        size_t max_index = 0;
        for (size_t i = 1; i < N; i++)
            if ((*this)[i] > (*this)[max_index])
                max_index = i;
        return max_index;
    }
    void display() const
    {
        for (auto i : *this)
            cout << i << " ";
        cout << '\n';
    }
};
// Two-dimensional sizes (unused)
struct Size2D
{
    size_t x, y;
    constexpr Size2D() : x(0), y(0) {}
    constexpr Size2D(size_t size) : x(size), y(size) {}
    constexpr Size2D(size_t x, size_t y) : x(x), y(y) {}
};
// Two-dimensional array
template <size_t M, size_t N>
struct Array2D : array<array<double, N>, M>
{
    template <typename Func>
    void for_each(Func &&f)
    {
        for (auto &i : *this)
            for (auto &j : i)
                f(j);
    }
    void zero()
    {
        for_each(cppnn::zero);
    }
    Array2D<M, N> &operator+=(const Array2D<M, N> &rhs)
    {
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++)
                (*this)[i][j] += rhs[i][j];
        return *this;
    }
    Array2D<M, N> &operator*=(const double rhs)
    {
        for_each([rhs](double &a) { a *= rhs; });
        return *this;
    }
    void display() const
    {
        for (auto &i : *this)
        {
            for (auto j : i)
                cout << j << " ";
            cout << '\n';
        }
    }
};
// M*N matrix
template <size_t M, size_t N>
struct Matrix : Array2D<M, N>
{
    Vec<M> operator*(const Vec<N> &v) const
    {
        Vec<M> ret;
        ret.zero();
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++)
                ret[i] += v[j] * (*this)[i][j];
        return ret;
    }
};
// Two-dimensional data
template <size_t H, size_t W>
struct Image : Array2D<H, W>
{
    double get(ssize_t h, ssize_t w, double def = 0) const
    {
        if (0 <= h && h < H && 0 <= w && w < W)
            return (*this)[h][w];
        else
            return def;
    }
    double &at(ssize_t h, ssize_t w)
    {
        if (0 <= h && h < H && 0 <= w && w < W)
            return (*this)[h][w];
        else
            throw out_of_range("Image index out of bound");
    }
    template <typename T>
    auto apply(const T &val) const
    {
        return val.forward(*this);
    }
    template <typename Func>
    auto apply() const
    {
        return Func::forward(*this);
    }
    Vec<H * W> flatten() const
    {
        Vec<H * W> v;
        for (size_t i = 0; i < H; i++)
            for (size_t j = 0; j < W; j++)
                v[i * H + j] = (*this)[i][j];
        return v;
    }
    static Image<H, W> reshape(const Vec<H * W> &v)
    {
        Image<H, W> img;
        for (size_t i = 0; i < H; i++)
            for (size_t j = 0; j < W; j++)
                img[i][j] = v[i * H + j];
        return img;
    }
};
// Array of two-dimensional data
template <size_t N, size_t H, size_t W>
struct Images : array<Image<H, W>, N>
{
    template <typename Func>
    void for_each(Func &&f)
    {
        for (auto &i : *this)
            i.for_each(f);
    }
    void zero()
    {
        for_each(cppnn::zero);
    }
    Images<N, H, W> &operator+=(const Images<N, H, W> &rhs)
    {
        for (size_t i = 0; i < N; i++)
            (*this)[i] += rhs[i];
        return *this;
    }
    Images<N, H, W> &operator*=(const double rhs)
    {
        for_each([rhs](double &a) { a *= rhs; });
        return *this;
    }
    template <typename T>
    auto apply(const T &val) const
    {
        return val.forward(*this);
    }
    template <typename Func>
    auto apply() const
    {
        return Func::forward(*this);
    }
    Vec<N * H * W> flatten() const
    {
        Vec<N * H * W> ret;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < H; j++)
                for (size_t k = 0; k < W; k++)
                    ret[i * H * W + j * W + k] = (*this)[i][j][k];
        return ret;
    }
    static Images<N, H, W> reshape(const Vec<N * H * W> &v)
    {
        Images<N, H, W> ret;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < H; j++)
                for (size_t k = 0; k < W; k++)
                    ret[i][j][k] = v[i * H * W + j * W + k];
        return ret;
    }
};
} // namespace cppnn

#endif // _CPPNN_DATA_STRUCTURE_HPP