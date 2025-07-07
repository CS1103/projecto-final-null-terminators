#include "utec/nn/neural_network.h"
#include <Eigen/Dense>
#include <cmath>
#include <fstream>

using namespace utec;
using namespace utec::algebra;
using namespace utec::nn;

NeuralNetwork::NeuralNetwork(std::size_t input_size, std::size_t num_classes) {
    W_ = Tensor::Random(input_size, num_classes);
    b_ = Tensor::Zero(1, num_classes);
}

static Tensor softmax(const Tensor& z) {
    Tensor exp_z = z.array().exp();
    Eigen::VectorXf sums = exp_z.rowwise().sum();
    return exp_z.array().colwise() / sums.array();
}

void NeuralNetwork::train(const Tensor& x, const Tensor& y,
                          std::size_t epochs, float lr) {
    for (std::size_t e = 0; e < epochs; ++e) {
        Tensor logits = (x * W_).rowwise() + b_.row(0);
        Tensor probs = softmax(logits);
        Tensor diff = probs - y;
        Tensor grad_W = x.transpose() * diff / x.rows();
        Tensor grad_b = diff.colwise().mean();
        W_ -= lr * grad_W;
        b_ -= lr * grad_b;
    }
}

Tensor NeuralNetwork::predict(const Tensor& x) const {
    Tensor logits = (x * W_).rowwise() + b_.row(0);
    return softmax(logits);
}

void NeuralNetwork::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    int rows = W_.rows();
    int cols = W_.cols();
    out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(int));
    out.write(reinterpret_cast<const char*>(W_.data()), rows*cols*sizeof(float));
    int bcols = b_.cols();
    out.write(reinterpret_cast<const char*>(&bcols), sizeof(int));
    out.write(reinterpret_cast<const char*>(b_.data()), bcols*sizeof(float));
}

void NeuralNetwork::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    int rows, cols, bcols;
    in.read(reinterpret_cast<char*>(&rows), sizeof(int));
    in.read(reinterpret_cast<char*>(&cols), sizeof(int));
    W_.resize(rows, cols);
    in.read(reinterpret_cast<char*>(W_.data()), rows*cols*sizeof(float));
    in.read(reinterpret_cast<char*>(&bcols), sizeof(int));
    b_.resize(1, bcols);
    in.read(reinterpret_cast<char*>(b_.data()), bcols*sizeof(float));
}
