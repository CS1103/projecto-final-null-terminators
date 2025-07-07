# Proyecto de Clasificación de Dígitos

Este repositorio contiene un ejemplo sencillo de red neuronal implementada en **C++** utilizando la biblioteca Eigen.

El ejecutable `digit_classifier` permite entrenar el modelo con un archivo CSV del dataset MNIST y clasificar imágenes JPG en escala de grises.

## Uso rápido

```bash
# Compilar
g++ -Iinclude -I/usr/include/eigen3 -std=c++17 src/stb_image_impl.cpp \
    src/utec/nn/neural_network.cpp src/digit_classifier.cpp -o digit_classifier

# Entrenar el modelo
./digit_classifier train mnist_train.csv

# Clasificar imágenes
./digit_classifier imagen1.jpg imagen2.jpg
```

El modelo entrenado se almacena en `models/digit_weights.bin`.
