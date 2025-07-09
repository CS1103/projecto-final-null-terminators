#pragma once
#include <vector>
#include <initializer_list>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <numeric>

namespace utec {
namespace algebra {

class Tensor {
public:
    // Constructores
    Tensor() = default;
    Tensor(const std::vector<size_t>& shape)
        : shape_(shape), data_(product(shape), 0.0f) {}
    Tensor(const std::vector<size_t>& shape, float value)
        : shape_(shape), data_(product(shape), value) {}
    Tensor(std::initializer_list<float> values)
        : shape_{values.size()}, data_(values) {}
    Tensor(std::initializer_list<size_t> dims)
        : shape_(dims),
          data_(std::accumulate(dims.begin(), dims.end(), size_t(1), [](size_t a, size_t b) { return a * b; }), 0.0f) {}
    Tensor(std::initializer_list<size_t> dims, float value)
        : shape_(dims),
          data_(std::accumulate(dims.begin(), dims.end(), size_t(1), [](size_t a, size_t b) { return a * b; }), value) {}

    // Acceso a datos
    float& operator[](size_t idx) { return data_[idx]; }
    const float& operator[](size_t idx) const { return data_[idx]; }

    float& at(const std::vector<size_t>& indices) {
        return data_[flatten_index(indices)];
    }
    const float& at(const std::vector<size_t>& indices) const {
        return data_[flatten_index(indices)];
    }

    // Operaciones básicas
    Tensor operator+(const Tensor& other) const {
        assert(shape_ == other.shape_);
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] + other.data_[i];
        return result;
    }
    Tensor operator-(const Tensor& other) const {
        assert(shape_ == other.shape_);
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] - other.data_[i];
        return result;
    }
    Tensor operator*(float scalar) const {
        Tensor result(shape_);
        for (size_t i = 0; i < data_.size(); ++i)
            result.data_[i] = data_[i] * scalar;
        return result;
    }
    // Producto punto para vectores
    float dot(const Tensor& other) const {
        assert(shape_ == other.shape_);
        float sum = 0.0f;
        for (size_t i = 0; i < data_.size(); ++i)
            sum += data_[i] * other.data_[i];
        return sum;
    }
    // Producto matricial (solo 2D)
    Tensor matmul(const Tensor& other) const {
        assert(shape_.size() == 2 && other.shape_.size() == 2);
        assert(shape_[1] == other.shape_[0]);
        Tensor result({shape_[0], other.shape_[1]});
        for (size_t i = 0; i < shape_[0]; ++i) {
            for (size_t j = 0; j < other.shape_[1]; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < shape_[1]; ++k) {
                    sum += at({i, k}) * other.at({k, j});
                }
                result.at({i, j}) = sum;
            }
        }
        return result;
    }

    // Utilidades
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }
    void fill(float value) { std::fill(data_.begin(), data_.end(), value); }
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }

private:
    std::vector<size_t> shape_;
    std::vector<float> data_;

    static size_t product(const std::vector<size_t>& dims) {
        size_t prod = 1;
        for (auto d : dims) prod *= d;
        return prod;
    }
    size_t flatten_index(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            std::cerr << "[Tensor::flatten_index] Shape: ";
            for (auto s : shape_) std::cerr << s << " ";
            std::cerr << " Indices: ";
            for (auto i : indices) std::cerr << i << " ";
            std::cerr << std::endl;
        }
        assert(indices.size() == shape_.size());
        size_t idx = 0, mult = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            idx += indices[i] * mult;
            mult *= shape_[i];
        }
        return idx;
    }
};

} // namespace algebra
} // namespace utec