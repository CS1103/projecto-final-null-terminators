#pragma once
#include <algorithm>
#include <cmath>
#include "utec/algebra/Tensor.h"
#include "utec/nn/layer.h"

namespace utec {
namespace nn {

enum class ActivationType { ReLU, Softmax };

class Activation : public Layer {
public:
    Activation(ActivationType type) : type_(type) {}

    algebra::Tensor forward(const algebra::Tensor& input) override {
        input_ = input;
        algebra::Tensor output = input;
        if (type_ == ActivationType::ReLU) {
            for (auto& v : output.data())
                v = std::max(0.0f, v);
        } else if (type_ == ActivationType::Softmax) {
            float max_val = *std::max_element(output.data().begin(), output.data().end());
            float sum = 0.0f;
            for (auto& v : output.data()) {
                v = std::exp(v - max_val);
                sum += v;
            }
            for (auto& v : output.data())
                v /= sum;
        }
        return output;
    }

    algebra::Tensor backward(const algebra::Tensor& grad_output, float /*lr*/) override {
        algebra::Tensor grad = grad_output;
        if (type_ == ActivationType::ReLU) {
            for (size_t i = 0; i < grad.size(); ++i)
                grad.data()[i] *= (input_.data()[i] > 0) ? 1.0f : 0.0f;
        } else if (type_ == ActivationType::Softmax) {
            // Aproximación: grad_output ya es el gradiente correcto para softmax+crossentropy
        }
        return grad;
    }

private:
    ActivationType type_;
    algebra::Tensor input_;
};

} // namespace nn
} // namespace utec
