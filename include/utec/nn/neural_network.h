#pragma once
#include <cstddef>
#include "../algebra/Tensor.h"
#include <string>

namespace utec {
namespace nn {

class NeuralNetwork {
public:
    NeuralNetwork(std::size_t input_size = 784, std::size_t num_classes = 10);
    void train(const algebra::Tensor& x, const algebra::Tensor& y,
               std::size_t epochs = 1, float lr = 0.01f);
    algebra::Tensor predict(const algebra::Tensor& x) const;
    void save(const std::string& path) const;
    void load(const std::string& path);
private:
    algebra::Tensor W_;
    algebra::Tensor b_;
};

} // namespace nn
} // namespace utec
