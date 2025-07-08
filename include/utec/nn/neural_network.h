#pragma once
#include <vector>
#include <memory>
#include "utec/algebra/Tensor.h"
#include "utec/nn/layer.h"
#include "utec/nn/dense.h"
#include "utec/nn/activation.h"
#include "utec/nn/loss.h"

namespace utec {
namespace nn {

class NeuralNetwork {
public:
    NeuralNetwork(size_t input_size, size_t num_classes) {
        layers_.emplace_back(std::make_unique<Dense>(input_size, 128));
        layers_.emplace_back(std::make_unique<Activation>(ActivationType::ReLU));
        layers_.emplace_back(std::make_unique<Dense>(128, 64));
        layers_.emplace_back(std::make_unique<Activation>(ActivationType::ReLU));
        layers_.emplace_back(std::make_unique<Dense>(64, num_classes));
        layers_.emplace_back(std::make_unique<Activation>(ActivationType::Softmax));
    }

    // Permite agregar capas personalizadas
    void add_layer(std::unique_ptr<Layer> layer) {
        layers_.emplace_back(std::move(layer));
    }

    algebra::Tensor forward(const algebra::Tensor& input) {
        algebra::Tensor x = input;
        for (auto& layer : layers_)
            x = layer->forward(x);
        return x;
    }

    void train_step(const algebra::Tensor& input, const algebra::Tensor& target, float lr) {
        std::vector<algebra::Tensor> activations = {input};
        for (auto& layer : layers_)
            activations.push_back(layer->forward(activations.back()));
        algebra::Tensor grad = loss::cross_entropy_derivative(activations.back(), target);
        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i)
            grad = layers_[i]->backward(grad, lr);
    }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};

} // namespace nn
} // namespace utec
