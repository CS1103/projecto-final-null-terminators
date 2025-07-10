#include <iostream>
#include <vector>
#include <string>
#include "utec/agent/DigitClassifier.h"
#include "utec/algebra/Tensor.h"
#include "utils/mnist_loader.h"
#include "lodepng.h"
#include <cmath>
#include <array>
#include <algorithm>
#include <random>
#include <numeric> // Required for std::iota

// Redimensiona una imagen RGBA a 28x28 usando nearest neighbor
std::vector<unsigned char> resize_rgba_to_28x28(const std::vector<unsigned char>& src, unsigned src_w, unsigned src_h) {
    std::vector<unsigned char> dst(28 * 28 * 4);
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            unsigned src_x = static_cast<unsigned>(x * src_w / 28.0);
            unsigned src_y = static_cast<unsigned>(y * src_h / 28.0);
            for (int c = 0; c < 4; ++c) {
                dst[4 * (y * 28 + x) + c] = src[4 * (src_y * src_w + src_x) + c];
            }
        }
    }
    return dst;
}

// Carga un PNG y lo convierte a un tensor 1D normalizado (0-1), redimensionando si es necesario
utec::algebra::Tensor load_png_as_tensor(const std::string& filename) {
    std::vector<unsigned char> image; // RGBA
    unsigned width, height;
    unsigned error = lodepng::decode(image, width, height, filename);
    if (error) {
        std::cerr << "Error al cargar PNG: " << lodepng_error_text(error) << std::endl;
        return utec::algebra::Tensor();
    }
    if (width != 28 || height != 28) {
        std::cout << "Redimensionando imagen a 28x28..." << std::endl;
        image = resize_rgba_to_28x28(image, width, height);
        width = height = 28;
    }
    std::vector<float> gray;
    for (size_t i = 0; i < image.size(); i += 4) {
        float r = image[i];
        float g = image[i+1];
        float b = image[i+2];
        float v = (r + g + b) / (3.0f * 255.0f);
        gray.push_back(v);
    }
    utec::algebra::Tensor tensor(std::vector<size_t>{width * height});
    for (size_t i = 0; i < gray.size(); ++i)
        tensor[i] = gray[i];
    return tensor;
}

int main() {
    // Cargar datos reales de MNIST
    std::cout << "Cargando datos de MNIST..." << std::endl;
    auto train_images = mnist_loader::load_mnist_images("data/train-images-idx3-ubyte");
    auto train_labels = mnist_loader::load_mnist_labels("data/train-labels-idx1-ubyte");
    std::cout << "Imágenes de entrenamiento: " << train_images.shape()[0] << " x " << train_images.shape()[1] << std::endl;
    std::cout << "Etiquetas de entrenamiento: " << train_labels.shape()[0] << " x " << train_labels.shape()[1] << std::endl;

    // Seleccionar numero de cada digito actual: 7000 (7k para la cantidad exacta de cada numero)
    std::vector<size_t> selected_indices;
    std::array<int, 10> count_per_digit = {0};
    size_t num_per_digit = 3000; // Trabajamos con 30k datos

    // Mezclar los índices para que la selección sea aleatoria
    std::vector<size_t> indices(train_labels.shape()[0]);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    for (size_t idx : indices) {
        // Buscar el dígito de la etiqueta
        int digit = -1;
        for (int d = 0; d < 10; ++d) {
            if (train_labels.at({idx, size_t(d)}) == 1.0f) {
                digit = d;
                break;
            }
        }
        if (digit != -1 && count_per_digit[digit] < num_per_digit) {
            selected_indices.push_back(idx);
            count_per_digit[digit]++;
        }
        // Si ya tenemos 100 de cada uno, salimos
        if (std::all_of(count_per_digit.begin(), count_per_digit.end(),
                        [num_per_digit](int c){ return c >= num_per_digit; }))
            break;
    }

    // Ahora crea los vectores de tensores solo con los seleccionados
    std::vector<utec::algebra::Tensor> train_images_vec, train_labels_vec;
    for (size_t idx : selected_indices) {
        utec::algebra::Tensor img(std::vector<size_t>{1, 28*28});
        for (size_t j = 0; j < 28*28; ++j)
            img.at({0, j}) = train_images.at({idx, j});
        train_images_vec.push_back(img);

        utec::algebra::Tensor lbl(std::vector<size_t>{1, 10});
        for (size_t j = 0; j < 10; ++j)
            lbl.at({0, j}) = train_labels.at({idx, j});
        train_labels_vec.push_back(lbl);
    }

    // (Opcional) Imprime cuántos hay de cada dígito
    std::array<int, 10> count_check = {0};
    for (const auto& lbl : train_labels_vec) {
        for (int d = 0; d < 10; ++d)
            if (lbl.at({0, size_t(d)}) == 1.0f)
                count_check[d]++;
    }
    for (int d = 0; d < 10; ++d)
        std::cout << "Dígito " << d << ": " << count_check[d] << std::endl;

    // 2. Entrenar la red
    utec::agent::DigitClassifier classifier(28*28, 10);
    std::cout << "Entrenando la red neuronal..." << std::endl;
    // Convertir los tensores a vectores de tensores para el API actual
    // std::vector<utec::algebra::Tensor> train_images_vec, train_labels_vec; // This block is now moved above
    // 3 epochs, 0.01 learning rate, 512 batch size
    classifier.train(train_images_vec, train_labels_vec, 3, 0.01f, 512); 
    std::cout << "Entrenamiento finalizado." << std::endl;

    // Detección automática de images/0m.png a images/9m.png 
    std::cout << "\n--- Prueba automática de images/0m.png a images/9m.png ---" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::string path = "images/" + std::to_string(i) + "m.png";
        auto img_tensor = load_png_as_tensor(path);
        if (img_tensor.size() != 28*28) {
            std::cerr << "No se pudo cargar o la imagen no es 28x28: " << path << std::endl;
            continue;
        }
        utec::algebra::Tensor img_batch(std::vector<size_t>{1, 28*28});
        for (size_t j = 0; j < 28*28; ++j)
            img_batch.at({0, j}) = img_tensor[j];
        size_t pred = classifier.predict(img_batch);
        std::cout << "Imagen " << path << " -> Predicción: " << pred << std::endl;
    }
    std::cout << "--- Fin de prueba automática ---\n" << std::endl;

    while (true) {
        std::string path;
        std::cout << "Ingrese la ruta de la imagen PNG (o 'salir' para terminar): ";
        std::cin >> path;
        if (path == "salir" || path == "exit") break;

        auto img_tensor = load_png_as_tensor(path);
        if (img_tensor.size() != 28*28) {
            std::cerr << "La imagen debe ser de 28x28 pixeles." << std::endl;
            continue;
        }
        utec::algebra::Tensor img_batch(std::vector<size_t>{1, 28*28});
        for (size_t i = 0; i < 28*28; ++i)
            img_batch.at({0, i}) = img_tensor[i];
        size_t pred = classifier.predict(img_batch);
        std::cout << "La red predice: " << pred << std::endl;
    }
    return 0;
}
