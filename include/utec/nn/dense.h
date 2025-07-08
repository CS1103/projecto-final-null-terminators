#pragma once
#include <random>
#include <memory>
#include "utec/algebra/Tensor.h"
#include "utec/nn/layer.h"

namespace utec {
namespace nn {

class Dense : public Layer {
public:
    Dense(size_t in_features, size_t out_features)
        : weights_({in_features, out_features}),
          bias_({out_features}),
          in_features_(in_features),
          out_features_(out_features)
    {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        for (auto& w : weights_.data()) w = dist(gen);
        for (auto& b : bias_.data()) b = dist(gen);
    }

    algebra::Tensor forward(const algebra::Tensor& input) override {
        input_ = input;
        algebra::Tensor out = input.matmul(weights_);
        for (size_t i = 0; i < out.size(); ++i)
            out.data()[i] += bias_.data()[i % out_features_];
        return out;
    }

    algebra::Tensor backward(const algebra::Tensor& grad_output, float lr) override {
        algebra::Tensor grad_input = grad_output.matmul(transpose(weights_));
        algebra::Tensor grad_weights = transpose(input_).matmul(grad_output);
        for (size_t i = 0; i < bias_.size(); ++i) {
            float grad = 0.0f;
            for (size_t j = 0; j < grad_output.size() / bias_.size(); ++j)
                grad += grad_output.data()[j * bias_.size() + i];
            bias_.data()[i] -= lr * grad;
        }
        for (size_t i = 0; i < weights_.size(); ++i)
            weights_.data()[i] -= lr * grad_weights.data()[i];
        return grad_input;
    }

private:
    algebra::Tensor weights_, bias_, input_;
    size_t in_features_, out_features_;

    algebra::Tensor transpose(const algebra::Tensor& t) {
        auto shape = t.shape();
        algebra::Tensor result({shape[1], shape[0]});
        for (size_t i = 0; i < shape[0]; ++i)
            for (size_t j = 0; j < shape[1]; ++j)
                result.at({j, i}) = t.at({i, j});
        return result;
    }
};

} // namespace nn
} // namespace utec
