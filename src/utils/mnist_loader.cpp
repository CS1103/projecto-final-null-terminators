#include "utils/mnist_loader.h"
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <algorithm>

namespace mnist_loader {

// Lee un entero de 4 bytes en big-endian
template<typename T=int>
T read_be(std::ifstream& f) {
    unsigned char bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    return (T(bytes[0]) << 24) | (T(bytes[1]) << 16) | (T(bytes[2]) << 8) | T(bytes[3]);
}

utec::algebra::Tensor load_mnist_images(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("No se pudo abrir archivo de imágenes MNIST: " + filename);
    int magic = read_be(f);
    if (magic != 2051) throw std::runtime_error("Archivo de imágenes MNIST inválido: " + filename);
    int num_images = read_be(f);
    int num_rows = read_be(f);
    int num_cols = read_be(f);
    if (num_rows != 28 || num_cols != 28) throw std::runtime_error("Solo se soportan imágenes 28x28");
    std::vector<float> data(num_images * 28 * 28);
    for (int i = 0; i < num_images * 28 * 28; ++i) {
        unsigned char pixel = 0;
        f.read(reinterpret_cast<char*>(&pixel), 1);
        data[i] = float(pixel) / 255.0f;
    }
    utec::algebra::Tensor tensor(std::vector<size_t>{size_t(num_images), 28*28});
    std::copy(data.begin(), data.end(), tensor.data().begin());
    return tensor;
}

utec::algebra::Tensor load_mnist_labels(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("No se pudo abrir archivo de etiquetas MNIST: " + filename);
    int magic = read_be(f);
    if (magic != 2049) throw std::runtime_error("Archivo de etiquetas MNIST inválido: " + filename);
    int num_labels = read_be(f);
    std::vector<float> data(num_labels * 10, 0.0f);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        f.read(reinterpret_cast<char*>(&label), 1);
        if (label < 10)
            data[i * 10 + label] = 1.0f;
    }
    utec::algebra::Tensor tensor(std::vector<size_t>{size_t(num_labels), 10});
    std::copy(data.begin(), data.end(), tensor.data().begin());
    return tensor;
}

} // namespace mnist_loader
