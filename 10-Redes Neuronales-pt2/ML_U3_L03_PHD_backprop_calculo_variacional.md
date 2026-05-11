# Backpropagation como Diferenciación Automática: Una Perspectiva Algebraica

**Unidad 3 · Lectura complementaria 03 · Audiencia: Doctorado**
versión: 2025-1 | modificado: 2026-05-09 | lectura estimada: 7 min

---

La presentación estándar de backpropagation en libros de texto es procedimental: dado un grafo de cómputo, propaga los deltas desde la salida hacia la entrada usando la regla de la cadena. Es correcta, pero oculta algo más profundo. Backpropagation es un caso especial de *diferenciación automática en modo reverso* —un algoritmo general para calcular gradientes en grafos de cómputo arbitrarios. Entender esta conexión no es un lujo académico: es lo que permite entender por qué frameworks como PyTorch o JAX funcionan como funcionan, y por qué pueden diferenciarse a través de bucles, condicionales, y estructuras de datos complejas.

## El Grafo de Cómputo como Objeto Matemático

Cualquier red neuronal —y en general, cualquier programa diferenciable— puede representarse como un grafo dirigido acíclico (DAG) donde cada nodo es una operación elemental y las aristas representan flujo de datos. Las variables de entrada son nodos hoja, la pérdida es el nodo raíz, y cada nodo interior tiene asociada una función $f_i$ y su derivada $f'_i$.

Formalmente, si la red computa $\mathcal{L} = f_L \circ f_{L-1} \circ \cdots \circ f_1(\mathbf{x})$, el gradiente respecto a cualquier parámetro $w$ es:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial f_L} \cdot \frac{\partial f_L}{\partial f_{L-1}} \cdots \frac{\partial f_k}{\partial w}$$

donde el producto es en el sentido del Jacobiano matricial cuando las operaciones son vectoriales.

## Modo Directo vs. Modo Reverso

Existen dos maneras de evaluar el producto de Jacobianos que compone el gradiente.

En el **modo directo** (*forward mode*), se propaga la derivada desde la entrada hacia la salida: se calcula $\frac{\partial f_1}{\partial w}$, luego $\frac{\partial f_2}{\partial w}$, y así hasta $\frac{\partial \mathcal{L}}{\partial w}$. El costo por parámetro es proporcional al número de operaciones del grafo. Para redes con millones de parámetros, esto es prohibitivo: requeriría un pase completo por la red para cada uno.

En el **modo reverso** (*reverse mode*), que es backpropagation, se calcula en orden inverso: primero $\frac{\partial \mathcal{L}}{\partial f_L}$, luego $\frac{\partial \mathcal{L}}{\partial f_{L-1}}$ usando el anterior, y así hasta $\frac{\partial \mathcal{L}}{\partial w}$ para todos los parámetros simultáneamente. El costo total es $O(|\text{red}|)$ —lineal en el número de operaciones— independientemente del número de parámetros.

Esta asimetría no es accidental. Para funciones con muchas entradas y una sola salida escalar (como la pérdida), el modo reverso es exactamente lo que se necesita. Para funciones con pocas entradas y muchas salidas, el modo directo sería preferible. El gradiente de una pérdida escalar respecto a todos los parámetros es, canónicamente, el caso de muchas entradas y una salida —de ahí que backpropagation sea la elección natural.

## Las Variables Adjuntas y el Álgebra de los Deltas

La notación $\delta^{(l)}$ usada en la clase tiene una interpretación algebraica precisa. Se llama la **variable adjunta** de la capa $l$, y se define como:

$$\bar{\mathbf{z}}^{(l)} := \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}$$

Las variables adjuntas satisfacen una relación de recurrencia que es exactamente la ecuación de backpropagation:

$$\bar{\mathbf{z}}^{(l)} = \left(\mathbf{W}^{(l+1)\top} \bar{\mathbf{z}}^{(l+1)}\right) \odot \sigma'(\mathbf{z}^{(l)})$$

Esta no es una fórmula que se "deduce" cada vez. Es la consecuencia directa de aplicar la regla de la cadena al grafo de cómputo, propagando las adjuntas de salida a entrada. Los frameworks modernos no implementan esta fórmula para cada arquitectura —implementan reglas locales de diferenciación para cada operación elemental (multiplicación matricial, ReLU, sigmoid, etc.) y las componen automáticamente.

## La Inicialización de Pesos y el Espectro del Jacobiano

Una de las consecuencias más importantes de la perspectiva algebraica es entender por qué la inicialización de los pesos importa.

El gradiente en la capa $l$ contiene el producto matricial $\mathbf{W}^{(L)\top} \mathbf{W}^{(L-1)\top} \cdots \mathbf{W}^{(l+1)\top}$. Por el teorema de Gelfand, la norma de este producto crece o decrece exponencialmente con el número de capas, dependiendo del valor singular máximo de cada $\mathbf{W}^{(k)}$.

Si los valores singulares son $> 1$: los gradientes explotan al retropropagar. Si los valores singulares son $< 1$: los gradientes se desvanecen.

La inicialización **Xavier/Glorot** (2010) establece varianzas iniciales de los pesos como $\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$, con el objetivo de mantener la varianza de las activaciones y gradientes aproximadamente constante a través de las capas —es decir, mantener el espectro del Jacobiano cerca de 1.

La inicialización **He** (2015), diseñada para ReLU, usa $\text{Var}(w) = \frac{2}{n_{\text{in}}}$, corrigiendo por el hecho de que ReLU anula exactamente la mitad de las neuronas en expectativa, reduciendo la varianza efectiva a la mitad.

```python
# [PhD] Verificación empírica del efecto de la inicialización
import numpy as np

def check_gradient_flow(n_layers=10, n_neurons=100, init='xavier', seed=42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_neurons, 1))  # activación inicial
    grad = rng.standard_normal((n_neurons, 1))  # gradiente inicial (desde la salida)

    act_norms = [np.linalg.norm(x)]
    grad_norms = [np.linalg.norm(grad)]

    for _ in range(n_layers):
        n_in = n_neurons
        if init == 'xavier':
            std = np.sqrt(2 / (n_in + n_neurons))
        elif init == 'he':
            std = np.sqrt(2 / n_in)
        elif init == 'naive':
            std = 0.01  # inicialización naive — demasiado pequeña
        else:
            std = 1.0

        W = rng.normal(0, std, (n_neurons, n_in))
        x = np.maximum(0, W @ x)                         # forward ReLU
        grad = (W.T @ grad) * (W @ rng.standard_normal((n_in, 1)) > 0)  # backward
        act_norms.append(np.linalg.norm(x))
        grad_norms.append(np.linalg.norm(grad))

    return act_norms, grad_norms

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for init, color in [('naive', 'tomato'), ('xavier', 'steelblue'), ('he', 'green')]:
    acts, grads = check_gradient_flow(n_layers=15, init=init)
    axes[0].plot(acts, label=init, color=color, linewidth=2)
    axes[1].plot(grads, label=init, color=color, linewidth=2)
axes[0].set_title('Norma de Activaciones por Capa'); axes[0].set_xlabel('Capa'); axes[0].legend()
axes[1].set_title('Norma del Gradiente por Capa'); axes[1].set_xlabel('Capa'); axes[1].legend()
axes[0].set_yscale('log'); axes[1].set_yscale('log')
plt.tight_layout(); plt.show()
```

La gráfica debería mostrar que la inicialización naive hace que los gradientes y activaciones colapsen rápidamente, mientras que Xavier y He los mantienen estables a través de las capas.

## Para reflexionar

1. Los frameworks de diferenciación automática (PyTorch, JAX) permiten diferenciar a través de bucles `for`, condicionales `if` y llamadas recursivas. ¿Qué representa el grafo de cómputo en esos casos? ¿Cómo cambia el grafo en cada paso de entrenamiento?

2. La inicialización He asume que ReLU anula exactamente el 50% de las neuronas en expectativa. ¿Bajo qué condiciones podría violarse este supuesto durante el entrenamiento? ¿Qué consecuencias tendría para el flujo de gradientes?

3. Batch normalization (Ioffe & Szegedy, 2015) fue motivada en parte por el problema del gradiente que desaparece. ¿Cómo modifica la normalización por batch la dinámica del espectro del Jacobiano de la red? ¿Puede pensarse en ella como una forma de "reinicialización" continua?

## Para ir más lejos

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536. doi:10.1038/323533a0 [Original]

- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS 2010*, 249–256. [Inicialización Xavier y análisis del espectro]

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers. *ICCV 2015*. doi:10.1109/ICCV.2015.123 [Inicialización He para ReLU]

- Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: a survey. *JMLR*, 18(1), 5595–5637. [Revisión completa de diferenciación automática] https://jmlr.org/papers/v18/17-468.html

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Cap. 6 (Redes profundas) y Cap. 8 (Optimización). https://www.deeplearningbook.org

---
*Lectura relacionada con la Clase ML_U3_C01 · Sección 4 (Backpropagation)*
*· Lab ML_U3_Lab01 · Assignment ML_A7*
