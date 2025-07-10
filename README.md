## Programación III (CS2013) - Laboratorio 1.02 - 2025 - 1

---
# 🧠 Proyecto Final 2025-1: AI Neural Network

El proyecto detalla el desarrollo de un agente de Inteligencia Artificial para jugar Pong utilizando C++. El proyecto se estructura en varios "Epics" o módulos clave, que incluyen la creación de una biblioteca genérica de álgebra, una red neuronal completa, un agente Pong basado en la red, y aspectos de paralelismo, entrenamiento y documentación.

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
-----

## 🚀 Para empezar

### Requisitos del Sistema

Para compilar y ejecutar este proyecto, necesitarás lo siguiente:

  * **Compilador C++20**: Asegúrate de tener una versión reciente de tu compilador preferido que soporte C++20 (por ejemplo, GCC 10+ o Clang 10+).
  * **CMake**: Indispensable para la gestión del proyecto y la compilación. Se requiere la versión 3.10 o superior.

### Compilación del Proyecto

Una vez que tengas los requisitos, compilar es pan comido:

```bash
mkdir build         # Crea un directorio para la compilación
cd build            # Navega al directorio 'build'
cmake ..            # Configura el proyecto usando CMake
make                # Compila el código fuente
```

Esto generará el ejecutable `digit_classifier` (o similar, dependiendo de tu `CMakeLists.txt`) dentro del directorio `build`.

### Preparación de los Datos

El corazón de nuestro clasificador son los datos MNIST. Así es como los obtienes y los preparas:

1.  **Descarga el Conjunto de Datos MNIST**:
    Dirígete a la página oficial de MNIST y descarga los siguientes archivos comprimidos:

      * `train-images-idx3-ubyte.gz`
      * `train-labels-idx1-ubyte.gz`

2.  **Descomprime y Organiza**:
    Asegúrate de que los archivos estén **descomprimidos** (sin la extensión `.gz`) y colócalos en el directorio `data/` de tu proyecto. Los nombres exactos de los archivos deben ser:

      * `train-images-idx3-ubyte`
      * `train-labels-idx1-ubyte`

3.  **Añade tus Propias Imágenes (Opcional)**:
    Si quieres ver cómo el clasificador se desempeña con tus propios dibujos, coloca imágenes de dígitos en formato PNG (por ejemplo, `0m.png`, `1m.png`, etc.) dentro del directorio `images/`. ¡El programa las redimensionará automáticamente a 28x28 si es necesario\!

-----

## 🏆 Cumplimiento del Proyecto 

Aunque el enunciado original del proyecto se centraba en un "Pong AI", la infraestructura y el desarrollo de la red neuronal y la biblioteca de álgebra implementadas cumplen con los objetivos fundamentales de los Epics del curso. A continuación, se detalla cómo cada Epic se aborda en este proyecto de clasificación de dígitos MNIST:

### Epic 1: Biblioteca Genérica de Álgebra (`utec::algebra::Tensor`)

**Contexto:** Este Epic se centra en la construcción de una base sólida para cualquier operación numérica compleja, esencial para el funcionamiento interno de una red neuronal. Nuestra implementación del `Tensor` busca replicar la versatilidad de librerías como NumPy en C++.

| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                                                  |
| :------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Constructores variádicos y `std::array<size_t, Rank>`          | La implementación actual utiliza `std::vector<size_t>` e `initializer_list<size_t>` para la forma. El acceso multidimensional se realiza a través de `at(const std::vector<size_t>& indices)` y acceso lineal con `operator[]`. La funcionalidad es similar, aunque la sintaxis variádica específica no está presente. |
| Acceso variádico `operator()`                                  | Sustituido por `at(const std::vector<size_t>& indices)`.                                                                                                                                                                                                                                                                 |
| `shape()`                                                      | Retorna el `std::vector<size_t>` que representa la forma.                                                                                                                                                                                                                                                                 |
| `reshape()` (preservando elementos)                            | La clase `Tensor` no tiene un método `reshape` explícito.                                                                                                                                                                                                                                                                 |
| `fill()`                                                       | Permite rellenar el tensor con un valor escalar.                                                                                                                                                                                                                                                                           |
| Operadores aritméticos (`+`, `-`, `*` escalar, `*` tensor)    | `+` y `-` implementados. `*` escalar implementado. El producto matricial se maneja con `matmul()`. El "broadcasting implícito" no está implementado de forma genérica en `operator*`, pero la multiplicación matricial lo maneja en el contexto de la red neuronal.                                                      |
| `transpose_2d()` (para Rank=2)                                 | Implementada una función `transpose` separada dentro de la clase `Dense` que utiliza para transponer tensores 2D.                                                                                                                                                                         |

---
### Epic 2: Red Neuronal Full (`utec::nn`)

**Contexto:** Este Epic se enfoca en la construcción del core de la inteligencia artificial: el framework de redes neuronales, incluyendo sus componentes esenciales como capas y funciones de activación/pérdida.

| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                                                    |
| :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ILayer` (ahora `Layer`) con `forward` y `backward`           | La clase `Layer` define la interfaz con los métodos `forward` y `backward` (que recibe el `learning_rate` para la actualización de pesos dentro de la capa).                                                                                                                                                            |
| `Dense` con `W`, `b`, `dW`, `db` y `last_x` (ahora `input_`)  | La clase `Dense` gestiona sus pesos (`weights_`), sesgos (`bias_`) y la entrada previa (`input_`) para el *backpropagation*. Las actualizaciones de `dW` y `db` se aplican directamente a `weights_` y `bias_` dentro del `backward` de la capa.                                                         |
| `ReLU` con `mask`                                              | La clase `Activation` implementa `ReLU` y maneja su derivada eficientemente. No usa una `mask` explícita, sino que compara `input_.data()[i] > 0` en el `backward`.                                                                                                                                          |
| `MSELoss` con `forward` y `backward`                           | Se implementó `cross_entropy` y `cross_entropy_derivative` en el namespace `utec::nn::loss` en lugar de `MSELoss`, ya que la entropía cruzada es más adecuada para problemas de clasificación multiclase como MNIST.                                                                                                |
| Clase `NeuralNetwork` (`add_layer`, `forward`, `backward`, `optimize`, `train`) | La clase `NeuralNetwork` permite construir la red con `add_layer`. `forward` propaga la entrada. `train_step` encapsula el `backward` y la actualización de pesos (que se delega a las capas individuales). El método `train` de `DigitClassifier` orquesta el entrenamiento sobre múltiples épocas e imágenes. |

---
### Epic 3: Agente (`utec::agent::DigitClassifier`)

**Contexto:** Originalmente concebido para un agente de Pong, este Epic se adapta para la creación de un clasificador de dígitos, demostrando cómo la red neuronal puede ser encapsulada y utilizada para una tarea específica de percepción.

| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                  |
| :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Clase `PongAgent` (adaptado a `DigitClassifier`)              | La clase `DigitClassifier` cumple el rol de `PongAgent`, pero para el dominio de clasificación de dígitos.                                                                                                                                                                                       |
| Recibir `State` (adaptado a `Tensor` de imagen)               | El método `predict` del `DigitClassifier` recibe un `utec::algebra::Tensor` que representa la imagen (equivalente al `State` del Pong).                                                                                                                                                                |
| Decidir acción (`act()`, adaptado a `predict()`)              | El método `predict` de `DigitClassifier` devuelve el índice de la clase predicha (el dígito), que es análogo a la "acción" en el contexto del Pong.                                                                                                                                                             |
| Bucle de simulación con `forward`                             | El `main.cpp` demuestra este bucle tanto para las imágenes de prueba automáticas como para el modo interactivo, donde `forward` (vía `predict`) es invocado para cada nueva entrada.                                                                                                                              |

---
### Epic 4: Paralelismo y CUDA Opcional

**Contexto:** Este Epic explora la optimización del rendimiento de la red neuronal a través de la computación paralela, incluyendo el uso de hilos o la aceleración por GPU (CUDA).

| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                  |
| :------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ThreadPool` y cola concurrente para inferencias             | Esta funcionalidad no se ha incluido en la implementación actual.                                                                                                                                                                                                                           |
| Soporte CUDA                                                   | La implementación es puramente en CPU.                                                                                                                                                                                                                                                      |

---
### Epic 5: Entrenamiento, Validación y Documentación

**Contexto:** Este Epic abarca las fases cruciales del ciclo de vida de un modelo de Machine Learning: cómo se entrena, cómo se evalúa su rendimiento y cómo se documenta para su comprensión y uso.

| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                  |
| :------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pipeline de entrenamiento básico                               | El `main.cpp` y la clase `DigitClassifier` implementan un ciclo de entrenamiento básico con los datos de MNIST.                                                                                                                                                                       |
| Serialización del modelo                                       | La capacidad de guardar y cargar el modelo entrenado no está implementada.                                                                                                                                                                                                                   |
| Validación (Conjunto separado)                                 | El código actual solo usa datos de entrenamiento; no hay un conjunto de validación separado.                                                                                                                                                                                                 |
| Documentación                                                  | Este `README.md` cumple con el objetivo de documentación, explicando la estructura, el uso y el cumplimiento de los requisitos.                                                                                                                                                                  |
