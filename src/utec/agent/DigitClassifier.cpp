#include "utec/agent/DigitClassifier.h"
#include <algorithm>
#include <cassert>
#include <iostream> // Added for printing progress

namespace utec {
namespace agent {

DigitClassifier::DigitClassifier(size_t input_size, size_t num_classes)
    : nn_(input_size, num_classes) {}

void DigitClassifier::train(const std::vector<algebra::Tensor>& images,
                            const std::vector<algebra::Tensor>& labels,
                            size_t epochs, float lr, size_t batch_size) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        for (size_t i = 0; i < images.size(); ++i) {
            // Forward para obtener la predicción
            auto pred = nn_.forward(images[i]);
            // Calcula la pérdida
            float loss = utec::nn::loss::cross_entropy(pred, labels[i]);
            epoch_loss += loss;
            // Backward y actualización
            nn_.train_step(images[i], labels[i], lr);

            // Imprime progreso cada 10000 muestras
            if ((i+1) % 1000 == 0)
                std::cout << "  Progreso: " << (i+1) << "/" << images.size() << std::endl;
        }
        epoch_loss /= images.size();
        std::cout << "Época " << (epoch+1) << "/" << epochs
                  << " - Pérdida promedio: " << epoch_loss << std::endl;
    }
}

size_t DigitClassifier::predict(const algebra::Tensor& image) {
    // Expect image.shape() == {1, input_size}
    assert(image.shape().size() == 2 && image.shape()[0] == 1);
    auto output = nn_.forward(image);
    return std::distance(output.data().begin(),
                         std::max_element(output.data().begin(), output.data().end()));
}

} // namespace agent
} // namespace utec
