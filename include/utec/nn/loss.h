#pragma once
#include <cmath>
#include "utec/algebra/Tensor.h"

namespace utec {
namespace nn {
namespace loss {

// Cross-entropy loss para clasificación
inline float cross_entropy(const algebra::Tensor& pred, const algebra::Tensor& target) {
    float loss = 0.0f;
    for (size_t i = 0; i < pred.size(); ++i)
        loss -= target.data()[i] * std::log(pred.data()[i] + 1e-8f);
    return loss;
}

// Derivada de cross-entropy respecto a la salida softmax
inline algebra::Tensor cross_entropy_derivative(const algebra::Tensor& pred, const algebra::Tensor& target) {
    algebra::Tensor grad(pred.shape());
    for (size_t i = 0; i < pred.size(); ++i)
        grad.data()[i] = pred.data()[i] - target.data()[i];
    return grad;
}

} // namespace loss
} // namespace nn
} // namespace utec
