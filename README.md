## Programación III (CS2013) - Laboratorio 1.02 - 2025 - 1

---
# 🧠 Proyecto Final 2025-1: AI Neural Network

Este proyecto es una inmersión profunda en el mundo del **Machine Learning y las Redes Neuronales**, implementadas completamente desde cero en **C++**. Nuestro objetivo principal es la clasificación de dígitos escritos a mano utilizando el famoso conjunto de datos **MNIST**.

---

## 💻 Desarrolladores

* **Franco Aedo Farge**
* **Fátima Villón Zárate** 
* **Josue Luna Rocha**

---

## 💡 Puntos Clave y Características Estelares

Aquí un vistazo rápido a lo que este proyecto ofrece:

| Característica                     | Descripción Detallada                                                                                                                                                                             |
|:-----------------------------------| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Biblioteca de Álgebra Genérica** | Una implementación robusta de tensores multidimensionales (`utec::algebra::Tensor`) al más puro estilo NumPy. Es el *cerebro* detrás de todas las operaciones matemáticas de la red. Soporta sumas, restas, multiplicación escalar y el crucial producto matricial. |
| **Framework de Redes Neuronales**  | Construido desde cero en `utec::nn`, este framework permite ensamblar redes neuronales capa por capa.                                                                                      |
| **Capas Modulares**                | Incluye `Dense` (capas totalmente conectadas) y `Activation` (con funciones **ReLU** y **Softmax**).                                                                                       |
| **Función de Pérdida**             | Utiliza la **Entropía Cruzada (`cross_entropy`)**, ideal para problemas de clasificación multiclase, junto con su derivada para el *backpropagation*.                                   |
| **Arquitectura Flexible**          | La clase `NeuralNetwork` te permite definir y entrenar arquitecturas complejas, manejando tanto el `forward` (inferencia) como el `train_step` (paso de entrenamiento).              |
| **Clasificador de Dígitos**        | Una clase de alto nivel (`utec::agent::DigitClassifier`) que abstrae la complejidad de la red para la tarea específica de clasificar dígitos MNIST.                                        |
| **Carga de Datos MNIST**           | Herramientas (`mnist_loader.h/.cpp`) para cargar eficientemente las imágenes y etiquetas del conjunto de datos MNIST en formato binario.                                                 |
| **Procesamiento PNG**              | Capacidad de cargar, redimensionar a 28x28 y normalizar imágenes PNG personalizadas, permitiéndote probar la red con tus propios dibujos.                                                 |

---

## 📂 Estructura del Proyecto

La organización del código está pensada para la modularidad y la claridad:

```
├── include/
│   ├── utec/
│   │   ├── agent/
│   │   │   └── DigitClassifier.h      // Define la interfaz del clasificador de dígitos
│   │   ├── algebra/
│   │   │   └── Tensor.h               // Implementación de la clase Tensor
│   │   └── nn/
│   │       ├── activation.h           // Funciones de activación (ReLU, Softmax)
│   │       ├── dense.h                // Capa densa (fully connected)
│   │       ├── layer.h                // Interfaz base para todas las capas
│   │       ├── loss.h                 // Funciones de pérdida (Cross-Entropy)
│   │   │   └── neural_network.h       // Clase principal de la red neuronal
│   └── utils/
│       ├── mnist_loader.h             // Utilidades para cargar datos MNIST
│       └── lodepng.h                  // Biblioteca para cargar PNG (tercero)
├── src/
│   ├── utec/
│   │   ├── agent/
│   │   │   └── DigitClassifier.cpp    // Implementación del clasificador de dígitos
│   │   └── main.cpp                   // Punto de entrada principal para entrenamiento y predicción
│   └── utils/
│       ├── mnist_loader.cpp           // Implementación de la carga de datos MNIST
│       └── lodepng.cpp                // Implementación de la biblioteca lodepng
├── data/                               // Directorio para los archivos de datos MNIST (ej: train-images-idx3-ubyte)
├── images/                             // Directorio para imágenes PNG de prueba (ej: 0m.png, 1m.png, etc.)
├── .gitignore
├── CMakeLists.txt                      // Archivo de configuración de CMake
└── README.md                           // Este archivo
```                         
---

## 🚀Cómo Empezar

### Requisitos

* **Compilador C++**
* **CMake**: Versión 3.10 o superior.

### Preparación de Datos

Antes de ejecutar, necesitas los datos MNIST:

1.  **Descarga los archivos MNIST**: Consíguelos desde la página oficial de MNIST:
    * `train-images-idx3-ubyte.gz`
    * `train-labels-idx1-ubyte.gz`
2.  **Descomprime**: Asegúrate de que los archivos estén descomprimidos (sin `.gz`) y colócalos en el directorio `data/` del proyecto. Los nombres de archivo deben ser `train-images-idx3-ubyte` y `train-labels-idx1-ubyte`.
3.  **Imágenes de Prueba (Opcional pero recomendado)**: Si deseas probar la funcionalidad de clasificación de PNG, coloca algunas imágenes de dígitos (por ejemplo, `0m.png` a `9m.png`) en el directorio `images/`.

---

## 🏆Cumplimiento de los epics

### Epic 1: ✨ Biblioteca Genérica de Álgebra (`utec::algebra::Tensor`)

| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                                                  |
| :------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Constructores variádicos y `std::array<size_t, Rank>`          | La implementación actual utiliza `std::vector<size_t>` e `initializer_list<size_t>` para la forma. El acceso multidimensional se realiza a través de `at(const std::vector<size_t>& indices)` y acceso lineal con `operator[]`. La funcionalidad es similar, aunque la sintaxis variádica específica no está presente. |
| Acceso variádico `operator()`                                  | Sustituido por `at(const std::vector<size_t>& indices)`.                                                                                                                                                                                                                                                                 |
| `shape()`                                                      | Retorna el `std::vector<size_t>` que representa la forma.                                                                                                                                                                                                                                                                 |
| `reshape()` (preservando elementos)                            | La clase `Tensor` no tiene un método `reshape` explícito.                                                                                                                                                                                                                                                                 |
| `fill()`                                                       | Permite rellenar el tensor con un valor escalar.                                                                                                                                                                                                                                                                           |
| Operadores aritméticos (`+`, `-`, `*` escalar, `*` tensor)    | `+` y `-` implementados. `*` escalar implementado. El producto matricial se maneja con `matmul()`. El "broadcasting implícito" no está implementado de forma genérica en `operator*`, pero la multiplicación matricial lo maneja en el contexto de la red neuronal.                                                      |
| `transpose_2d()` (para Rank=2)                                 | Implementada una función `transpose` separada dentro de la clase `Dense` que utiliza para transponer tensores 2D.                                                                                                                                                                                                        |

---
### Epic 2: ✨ Red Neuronal Full (`utec::nn`)

| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                                                    |
| :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ILayer` (ahora `Layer`) con `forward` y `backward`           | La clase `Layer` define la interfaz con los métodos `forward` y `backward` (que recibe el `learning_rate` para la actualización de pesos dentro de la capa).                                                                                                                                                            |
| `Dense` con `W`, `b`, `dW`, `db` y `last_x` (ahora `input_`)  | La clase `Dense` gestiona sus pesos (`weights_`), sesgos (`bias_`) y la entrada previa (`input_`) para el *backpropagation*. Las actualizaciones de `dW` y `db` se aplican directamente a `weights_` y `bias_` dentro del `backward` de la capa.                                                         |
| `ReLU` con `mask`                                              | La clase `Activation` implementa `ReLU` y maneja su derivada eficientemente. No usa una `mask` explícita, sino que compara `input_.data()[i] > 0` en el `backward`.                                                                                                                                          |
| `MSELoss` con `forward` y `backward`                           | Se implementó `cross_entropy` y `cross_entropy_derivative` en el namespace `utec::nn::loss` en lugar de `MSELoss`, ya que la entropía cruzada es más adecuada para problemas de clasificación multiclase como MNIST.                                                                                                |
| Clase `NeuralNetwork` (`add_layer`, `forward`, `backward`, `optimize`, `train`) | La clase `NeuralNetwork` permite construir la red con `add_layer`. `forward` propaga la entrada. `train_step` encapsula el `backward` y la actualización de pesos (que se delega a las capas individuales). El método `train` de `DigitClassifier` orquesta el entrenamiento sobre múltiples épocas e imágenes. |

---
### Epic 3: ✨ Agente (`utec::agent::DigitClassifier`)
| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                  |
| :------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Clase `PongAgent` (adaptado a `DigitClassifier`)              | La clase `DigitClassifier` cumple el rol de `PongAgent`, pero para el dominio de clasificación de dígitos.                                                                                                                                                                                       |
| Recibir `State` (adaptado a `Tensor` de imagen)               | El método `predict` del `DigitClassifier` recibe un `utec::algebra::Tensor` que representa la imagen (equivalente al `State` del Pong).                                                                                                                                                                |
| Decidir acción (`act()`, adaptado a `predict()`)              | El método `predict` de `DigitClassifier` devuelve el índice de la clase predicha (el dígito), que es análogo a la "acción" en el contexto del Pong.                                                                                                                                                             |
| Bucle de simulación con `forward`                             | El `main.cpp` demuestra este bucle tanto para las imágenes de prueba automáticas como para el modo interactivo, donde `forward` (vía `predict`) es invocado para cada nueva entrada.                                                                                                                              |

---
### Epic 4: ✨ Paralelismo y CUDA Opcional

| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                  |
| :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ThreadPool` y cola concurrente para inferencias             | Esta funcionalidad no se ha incluido en la implementación actual.                                                                                                                                                                                                                           |
| Soporte CUDA                                                   | La implementación es puramente en CPU.                                                                                                                                                                                                                                                      |

---
### Epic 5: ✨ Entrenamiento, Validación y Documentación

| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                  |
| :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Pipeline de entrenamiento básico                               | El `main.cpp` y la clase `DigitClassifier` implementan un ciclo de entrenamiento básico con los datos de MNIST.                                                                                                                                                                       |
| Serialización del modelo                                       | La capacidad de guardar y cargar el modelo entrenado no está implementada.                                                                                                                                                                                                                   |
| Validación (Conjunto separado)                                 | El código actual solo usa datos de entrenamiento; no hay un conjunto de validación separado.                                                                                                                                                                                                 |
| Documentación                                                  | Este `README.md` cumple con el objetivo de documentación, explicando la estructura, el uso y el cumplimiento de los requisitos. 

---



