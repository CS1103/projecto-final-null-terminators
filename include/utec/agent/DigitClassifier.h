#pragma once
#include <vector>
#include "utec/algebra/Tensor.h"
#include "utec/nn/neural_network.h"

namespace utec {
namespace agent {

class DigitClassifier {
public:
    DigitClassifier(size_t input_size, size_t num_classes);
    void train(const std::vector<algebra::Tensor>& images,
               const std::vector<algebra::Tensor>& labels,
               size_t epochs, float lr, size_t batch_size = 32);
    size_t predict(const algebra::Tensor& image);
private:
    nn::NeuralNetwork nn_;
};

} // namespace agent
} // namespace utec
