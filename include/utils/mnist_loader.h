#pragma once
#include <string>
#include "utec/algebra/Tensor.h"

namespace mnist_loader {
// Carga imágenes MNIST y devuelve un Tensor de shape [num_images, 28*28] (valores normalizados 0-1)
utec::algebra::Tensor load_mnist_images(const std::string& filename);
// Carga etiquetas MNIST y devuelve un Tensor de shape [num_images, 10] (one-hot)
utec::algebra::Tensor load_mnist_labels(const std::string& filename);
} // namespace mnist_loader
