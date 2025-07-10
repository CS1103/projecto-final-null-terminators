# Proyecto Final 2025-1: AI Neural Network                                                                                                                                                                                    
CS2013 Programación III · Informe Final

---

## 🎮 Descripción

Implementación de una red neuronal multicapa en C++ para clasificación de dígitos manuscritos, incluyendo una biblioteca de álgebra genérica, un framework de redes neuronales modular, y herramientas para carga de datos MNIST y procesamiento de imágenes PNG.

---

## 👥 Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: Null_Terminators
* **Desarrolladores**:
    * Franco Aedo Farge 
    * Fátima Villón Zárate 
    * Josue Luna Rocha 
* Este proyecto es el **Laboratorio 1.02 de Programación III (CS2013)** del ciclo **2025-1.**

---

## 💻 Requisitos e instalación

* **Compilador**: C++20 o superior
* **Dependencias**:
    * CMake 3.10+
    * Biblioteca `lodepng` (incluida en el proyecto)
* **Instalación**:
    ```bash
    mkdir build        # Crea un directorio para la compilación
    cd build           # Navega al directorio 'build'
    cmake ..           # Configura el proyecto usando CMake
    make               # Compila el código fuente
    ```
  Esto generará el ejecutable `digit_classifier` (o similar, dependiendo de tu `CMakeLists.txt`) dentro del directorio `build`.


### 💾 Preparación de los Datos

1.  **Descarga el Conjunto de Datos MNIST**: Dirígete a la página oficial de MNIST y descarga los siguientes archivos comprimidos:
    * `train-images-idx3-ubyte.gz`
    * `train-labels-idx1-ubyte.gz`
2.  **Descomprime y Organiza**: Asegúrate de que los archivos estén **descomprimidos** (sin la extensión `.gz`) y colócalos en el directorio `data/` de tu proyecto. Los nombres exactos de los archivos deben ser:
    * `train-images-idx3-ubyte`
    * `train-labels-idx1-ubyte`
3.  **Añade tus Propias Imágenes (Opcional)**: Si quieres ver cómo el clasificador se desempeña con tus propios dibujos, coloca imágenes de dígitos en formato PNG (por ejemplo, `0m.png`, `1m.png`, etc.) dentro del directorio `images/`. ¡El programa las redimensionará automáticamente a 28x28 si es necesario!

---

## 🧩 Investigación teórica

**Objetivo**: Comprender fundamentos y arquitecturas clave de redes neuronales (NNs).

### Historia y evolución de las redes neuronales (NNs)

Las NNs, inspiradas en el cerebro, evolucionaron desde el modelo de **McCulloch y Pitts (1943)** y la **Regla de Hebb (1949)**. El **Perceptrón de Rosenblatt (1957)** fue pionero, pero las críticas de **Minsky y Papert (1969)** causaron un "invierno de IA". El campo resurgió en los 80 con la popularización de la **retropropagación (Werbos 1974, Rumelhart, Hinton, Williams 1986)**, consolidando las NNs como el núcleo de la **IA moderna** en **visión por computadora**, **PNL** y **robótica** [5].

### Principales arquitecturas de redes neuronales

* **a) MLP (Perceptrón Multicapa)**
  Red *feedforward* con capas ocultas, resuelve problemas **no lineales**. Aprende vía **retropropagación**. Es un **aproximador universal** de funciones continuas, pero su entrenamiento puede ser complejo [6].

* **b) CNN (Redes Neuronales Convolucionales)**
  Especializadas en **datos espaciales (imágenes)**. Usan **filtros convolucionales** para extraer características jerárquicas, capas de *pooling* y densas. Reducen parámetros, ideales para **visión por computadora** [3].

* **c) RNN (Redes Neuronales Recurrentes)**
  Diseñadas para **datos secuenciales**. Tienen **conexiones cíclicas** y **"memoria"** interna. Comparten parámetros, manejan secuencias de longitud variable, cruciales en **PNL** y **reconocimiento de voz** [2].

### Algoritmos de entrenamiento

* **a) Retropropagación**
  Algoritmo fundamental para entrenar NNs multicapa. Minimiza el **error** ajustando **pesos** usando **gradiente descendente**. Calcula derivadas parciales del error, propagando la información **hacia atrás** desde la salida, requiriendo **funciones de activación diferenciables** y una **tasa de aprendizaje** adecuada [4].

* **b) Optimizadores**
  Algoritmos para **ajustar parámetros** y minimizar la **función de pérdida**. Incluyen:
    * **Descenso de Gradiente**: Clásico, computacionalmente costoso.
    * **SGD (Descenso de Gradiente Estocástico)**: Más eficiente con lotes de datos.
    * **Momento**: Acelera la convergencia de SGD.
    * **Optimizadores adaptativos (Adagrad, RMSprop)**: Ajustan dinámicamente la **tasa de aprendizaje** por parámetro.
      La elección depende de los datos y el modelo [1].

---

## 🙌🏻 Diseño e implementación

### Arquitectura de la solución

El proyecto sigue una arquitectura modular orientada a objetos en C++, aunque no se mencionan explícitamente patrones de diseño específicos como Factory o Strategy, la estructura de clases como `Layer`, `Dense`, `Activation`, `NeuralNetwork` y `DigitClassifier` refleja un diseño claro y extensible.

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



## 🏆Cumplimiento de los epics

### Epic 1: Biblioteca Genérica de Álgebra (`utec::algebra::Tensor`)
Este Epic se centra en la construcción de una base sólida para cualquier operación numérica compleja, esencial para el funcionamiento interno de una red neuronal. Nuestra implementación del `Tensor` busca replicar la versatilidad de librerías como NumPy en C++.

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
### Epic 2: Red Neuronal Full (`utec::nn`)
Este Epic se enfoca en la construcción del core de la inteligencia artificial: el framework de redes neuronales, incluyendo sus componentes esenciales como capas y funciones de activación/pérdida.

| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                                                    |
| :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ILayer` (ahora `Layer`) con `forward` y `backward`           | La clase `Layer` define la interfaz con los métodos `forward` y `backward` (que recibe el `learning_rate` para la actualización de pesos dentro de la capa).                                                                                                                                                            |
| `Dense` con `W`, `b`, `dW`, `db` y `last_x` (ahora `input_`)  | La clase `Dense` gestiona sus pesos (`weights_`), sesgos (`bias_`) y la entrada previa (`input_`) para el *backpropagation*. Las actualizaciones de `dW` y `db` se aplican directamente a `weights_` y `bias_` dentro del `backward` de la capa.                                                         |
| `ReLU` con `mask`                                              | La clase `Activation` implementa `ReLU` y maneja su derivada eficientemente. No usa una `mask` explícita, sino que compara `input_.data()[i] > 0` en el `backward`.                                                                                                                                          |
| `MSELoss` con `forward` y `backward`                           | Se implementó `cross_entropy` y `cross_entropy_derivative` en el namespace `utec::nn::loss` en lugar de `MSELoss`, ya que la entropía cruzada es más adecuada para problemas de clasificación multiclase como MNIST.                                                                                                |
| Clase `NeuralNetwork` (`add_layer`, `forward`, `backward`, `optimize`, `train`) | La clase `NeuralNetwork` permite construir la red con `add_layer`. `forward` propaga la entrada. `train_step` encapsula el `backward` y la actualización de pesos (que se delega a las capas individuales). El método `train` de `DigitClassifier` orquesta el entrenamiento sobre múltiples épocas e imágenes. |

---
### Epic 3: Agente (`utec::agent::DigitClassifier`)
Originalmente concebido para un agente de Pong, este Epic se adapta para la creación de un clasificador de dígitos, demostrando cómo la red neuronal puede ser encapsulada y utilizada para una tarea específica de percepción.

| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                  |
| :------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Clase `PongAgent` (adaptado a `DigitClassifier`)              | La clase `DigitClassifier` cumple el rol de `PongAgent`, pero para el dominio de clasificación de dígitos.                                                                                                                                                                                       |
| Recibir `State` (adaptado a `Tensor` de imagen)               | El método `predict` del `DigitClassifier` recibe un `utec::algebra::Tensor` que representa la imagen (equivalente al `State` del Pong).                                                                                                                                                                |
| Decidir acción (`act()`, adaptado a `predict()`)              | El método `predict` de `DigitClassifier` devuelve el índice de la clase predicha (el dígito), que es análogo a la "acción" en el contexto del Pong.                                                                                                                                                             |
| Bucle de simulación con `forward`                             | El `main.cpp` demuestra este bucle tanto para las imágenes de prueba automáticas como para el modo interactivo, donde `forward` (vía `predict`) es invocado para cada nueva entrada.                                                                                                                              |

---
### Epic 4: Paralelismo y CUDA Opcional
Este Epic explora la optimización del rendimiento de la red neuronal a través de la computación paralela, incluyendo el uso de hilos o la aceleración por GPU (CUDA).

| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                  |
| :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ThreadPool` y cola concurrente para inferencias             | Esta funcionalidad no se ha incluido en la implementación actual.                                                                                                                                                                                                                           |
| Soporte CUDA                                                   | La implementación es puramente en CPU.                                                                                                                                                                                                                                                      |

---
### Epic 5: Entrenamiento, Validación y Documentación
Este Epic abarca las fases cruciales del ciclo de vida de un modelo de Machine Learning: cómo se entrena, cómo se evalúa su rendimiento y cómo se documenta para su comprensión y uso.

| Requisito del Enunciado                                        | Detalles de Implementación                                                                                                                                                                                                                                                                  |
| :------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Pipeline de entrenamiento básico                               | El `main.cpp` y la clase `DigitClassifier` implementan un ciclo de entrenamiento básico con los datos de MNIST.                                                                                                                                                                       |
| Serialización del modelo                                       | La capacidad de guardar y cargar el modelo entrenado no está implementada.                                                                                                                                                                                                                   |
| Validación (Conjunto separado)                                 | El código actual solo usa datos de entrenamiento; no hay un conjunto de validación separado.                                                                                                                                                                                                 |
| Documentación                                                  | Este `README.md` cumple con el objetivo de documentación, explicando la estructura, el uso y el cumplimiento de los requisitos. 

---

## 🚀 Manual de Uso y Casos de Prueba

Después de la compilación, el ejecutable `digit_classifier` estará disponible en el directorio `build`. Este programa permite entrenar la red neuronal con el conjunto de datos MNIST y realizar predicciones.

Para ejecutar el entrenamiento y la predicción, usarás:
```bash
./build/digit_classifier
```

El programa carga automáticamente los datos de MNIST, entrena la red (mostrando progreso y pérdida promedio por época) y clasifica imágenes PNG presentes en el directorio `images/`.

### Casos de prueba

* **Carga de Datos MNIST**: Verificar la correcta carga y preprocesamiento de imágenes y etiquetas.
* **Funcionalidad de Tensor**: Validar operaciones básicas (suma, resta, multiplicación escalar, producto matricial).
* **Forward y Backward Pass**: Probar la propagación hacia adelante y hacia atrás en capas `Dense` y `Activation` para asegurar el flujo de datos y cálculo de gradientes.
* **Convergencia del Entrenamiento**: Observar la reducción de la pérdida promedio por época para confirmar el aprendizaje de la red.
* **Clasificación de Imágenes Personalizadas**: Evaluar la capacidad del clasificador para predecir dígitos en imágenes PNG (redimensionadas y normalizadas).

---

##  🔎 Análisis del Rendimiento

### Métricas de Evaluación

* **Progreso de Entrenamiento**: Se monitorea el avance por **épocas**, con reportes de progreso cada **1000 muestras** procesadas.
* **Tiempo de Entrenamiento**: Actualmente **no se mide explícitamente**. Se considera integrar un temporizador para el registro futuro.
* **Pérdida Promedio**: Se calcula y reporta la **pérdida promedio por época** utilizando la función de **entropía cruzada**.
* **Precisión de Clasificación**: La **precisión final no se calcula** sobre un conjunto de validación, lo que representa una potencial mejora para una evaluación más robusta.

### Fortalezas y Limitaciones

* **Fortalezas**:
    * **Implementación Pura C++**: Código desarrollado desde cero con **dependencias mínimas**, facilitando la comprensión interna.
    * **Arquitectura Modular**: Diseño claro y extensible para futuras adiciones.
* **Limitaciones**:
    * **Ejecución Exclusiva en CPU**: Rendimiento **limitado por CPU**, restringiendo la escalabilidad a grandes datasets.
    * **Ausencia de Paralelización**: No se ha implementado **paralelismo (Epic 4)**, impidiendo una aceleración significativa.
    * **Falta de Conjunto de Validación**: Impide una evaluación objetiva de la **generalización** y detección de **sobreajuste (overfitting)**.


### Mejoras Futuras

* **Optimización de Álgebra Lineal**: Integrar bibliotecas como **BLAS** para acelerar **operaciones tensoriales** (ej. producto matricial).
* **Paralelización del Entrenamiento**: Implementar **paralelismo por lotes** (e.g., ThreadPools) o soporte **CUDA** para **aceleración por GPU**.
* **Persistencia del Modelo**: Añadir funcionalidad para **guardar y cargar modelos entrenados (serialización)**.
* **Evaluación Robusta**: Incorporar un **conjunto de validación dedicado** para medir la **generalización** y monitorizar el **overfitting**.

---

## 💯 Trabajo en equipo

| Tarea                      | Miembro            | Rol                               |
| :------------------------- | :----------------- |:----------------------------------|
| Investigación teórica      | Franco Aedo Farge  | Contribución general al proyecto. |
| Diseño de la arquitectura  | Fátima Villón Zárate | Contribución general al proyecto. |
| Implementación del modelo  | Josue Luna Rocha   | Contribución general al proyecto. |
| Documentación              | Equipo             | Creación de `README.md`.          |

---

## ✨ Conclusiones

**Logros**:
* Se implementó una biblioteca de álgebra genérica (`utec::algebra::Tensor`) para operaciones tensoriales fundamentales.
* Se construyó un framework de redes neuronales (`utec::nn`) modular desde cero, incluyendo capas (`Dense`, `Activation`) y funciones de pérdida (`cross_entropy`), permitiendo la creación de arquitecturas complejas.
* El `DigitClassifier` demuestra la aplicación de la red para la clasificación de dígitos manuscritos del conjunto de datos MNIST, con capacidad de carga de datos e inferencia.

**Aprendizajes**:
* Profundización en el algoritmo de *backpropagation* y la optimización de pesos y sesgos.
* Comprensión de la modularidad en el diseño de frameworks de Machine Learning.
* Manejo de estructuras de datos complejas (tensores multidimensionales) en C++.

---
## 🔝 Recomendaciones
Para futuras iteraciones, se recomienda explorar la implementación de paralelismo (multi-threading o CUDA) para mejorar el rendimiento, integrar un conjunto de validación para una evaluación más robusta del modelo, y desarrollar funcionalidades de serialización para guardar y cargar modelos entrenados.

---

## 📚 Bibliografía

[1] Analytics Vidhya. (2021, octubre 7). *Guía completa sobre optimizadores en aprendizaje profundo*. Recuperado de https://www-analyticsvidhya-com.translate.goog/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc

[2] C. Arana, *Redes neuronales recurrentes: Análisis de los modelos especializados en datos secuenciales* (Serie Documentos de Trabajo No. 797). Universidad del Centro de Estudios Macroeconómicos de Argentina (UCEMA), 2021. Recuperado de https://www.econstor.eu/handle/10419/238422

[3] Departamento de Matemática Aplicada. (2021). *Redes neuronales convoluciones – Introducción al aprendizaje automático*. Universidad Politécnica de Madrid. Recuperado de https://dcain.etsin.upm.es/~carlos/bookAA/05.7_RRNN_Convoluciones_CIFAR_10_INFORMATIVO.html

[4] A. Fraga Fatás, *El algoritmo de retropropagación en redes neuronales multicapa*. Trabajo de Fin de Grado, Universidad de Zaragoza, 2023. Recuperado de https://zaguan.unizar.es/record/154704/files/TAZ-TFG-2023-2967.pdf

[5] N. Laredo, *Redes Neuronales*. Tecnológico Nacional de México, Campus Laredo, s.f. Recuperado de https://nlaredo.tecnm.mx/takeyas/Apuntes/Inteligencia%20Artificial/Apuntes/tareas_alumnos/RNA/Redes%20Neuronales2.pdf

[6] Universidad de Sevilla. (s.f.). *Perceptrón multicapa*. Biblus. Recuperado de https://biblus.us.es/bibing/proyectos/abreproy/12166/fichero/Volumen+1+-+Memoria+descriptiva+del+proyecto%252F3+-+Perceptron+multicapa.pdf








