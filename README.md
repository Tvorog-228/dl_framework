# MyDL Framework: Deep Learning desde Cero

Este proyecto es una implementación educativa de un **Framework de Deep Learning** construido completamente desde cero utilizando **Python y NumPy**. El objetivo principal es desmitificar las "cajas negras" de las librerías modernas (como PyTorch) mediante la construcción de sus componentes fundamentales.

## Características del Framework

Mi motor de Deep Learning incluye:
* **Motor de Autograd**: Un sistema de diferenciación automática que rastrea las operaciones para calcular gradientes de forma recursiva.
* **Abstracción de Capas**: Arquitectura modular con clases como `Linear`, `Sequential`, `RNNCell` y `LSTMCell`.
* **Funciones de Activación**: Implementaciones de `Sigmoid`, `Tanh` y `Softmax`.
* **Optimizadores**: Gradiente Descendente Estocástico (SGD) con soporte para actualización de parámetros.
* **Procesamiento de Datos**: Utilidades para cargar, normalizar y realizar el "shuffling" de datasets complejos.

---

## Ejemplo 1: Generación de Texto (Shakespeare)

En este ejemplo, el modelo intenta aprender y simular el estilo literario de **William Shakespeare**.

* **El Reto**: Generar texto coherente prediciendo el lenguaje carácter por carácter.
* **Arquitectura**: Utiliza una **RNN (Red Neuronal Recurrente)** que mantiene una "memoria" de los caracteres anteriores para entender el contexto.
* **Funcionamiento**: 
    1. Procesa el archivo `Shakespear.txt`.
    2. Convierte cada carácter en un vector matemático (*Embedding*).
    3. Predice la probabilidad del siguiente carácter basándose en la secuencia previa.
    4. Permite ajustar la **Temperatura** para controlar qué tan "creativo" o determinista es el autor artificial.

> **Nota**: El modelo evoluciona de escribir ruido aleatorio a formar palabras y estructuras de diálogo dramático tras varias épocas de entrenamiento.

---

## Ejemplo 2: Detección de Neumonía en Rayos X

Aplicación del framework en el campo de la **Visión por Computadora** para el diagnóstico médico.

* **Dataset**: Basado en el dataset de Kaggle con ~5,800 imágenes de radiografías de tórax (Normal vs. Pneumonia).
* **Procesamiento de Imágenes**:
    * Redimensión de imágenes a $64 \times 64$ píxeles.
    * Normalización de píxeles al rango $[0, 1]$.
* **Arquitectura**: Un Perceptrón Multicapa (MLP) diseñado para clasificación binaria.
