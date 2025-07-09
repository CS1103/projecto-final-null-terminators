#pragma once
#include "utec/algebra/Tensor.h"

namespace utec {
namespace nn {

class Layer {
public:
    virtual algebra::Tensor forward(const algebra::Tensor& input) = 0;
    virtual algebra::Tensor backward(const algebra::Tensor& grad_output, float lr) = 0;
    virtual ~Layer() = default;
};

} // namespace nn
} // namespace utec
