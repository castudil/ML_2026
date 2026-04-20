# Impureza, Gini, Entropía y Conexión con la Teoría de la Información

**Unidad 2 · Lectura complementaria 12 · Audiencia: Doctorado**

versión: 2025-1 | modificado: 2026-04-19 | lectura estimada: 15 min

---

¿Por qué Gini y Entropía producen casi siempre el mismo árbol si son funciones matemáticamente distintas? La respuesta sorprenderá a quien espera encontrar una diferencia fundamental. La verdad es más sutil: ambas son casos especiales de una familia más amplia de medidas de impureza, y dentro de esa familia, la diferencia entre ellas es pequeña donde importa—en las divisiones cercanas al punto de equilibrio, donde ambas distinguen bien entre puros e impuros. La pregunta correcta no es "¿cuál es mejor?" sino "¿qué estructura matemática unifica ambas?"

## La familia de medidas de impureza de Tsallis

Cualquier medida de impureza razonable debe satisfacer un conjunto de axiomas. Primero, debe ser **máxima** en la distribución uniforme: si tenemos $K$ clases equiprobables, la impureza es máxima cuando $p_1 = p_2 = \cdots = p_K = 1/K$. Segundo, debe ser **mínima** en una distribución degenerada: si $p_1 = 1$ y $p_i = 0$ para $i > 1$, la impureza es cero. Tercero, debe ser **simétrica** en el permutación de las probabilidades: el orden no importa. Cuarto, debe ser **continua** en esas probabilidades.

Estos axiomas definen una clase entera de funciones. Gini e Impureza de Misclassification son casos especiales. Pero existe una generalización más profunda: la **entropía de Tsallis**, introducida por Constantino Tsallis en 1988 en el contexto de mecánica estadística. Se define como:

$$H_q(p) = \frac{1 - \sum_{k=1}^{K} p_k^q}{q - 1}$$

para $q > 0, q \neq 1$. Esta familia es paramétrica en $q$. Cuando $q \to 1$, recuperamos la Entropía de Shannon:

$$H_1(p) = -\sum_{k=1}^{K} p_k \log p_k$$

Cuando $q = 2$, obtenemos el índice de Gini:

$$H_2(p) = 1 - \sum_{k=1}^{K} p_k^2$$

Ambas satisfacen los axiomas. Ambas se usan en árboles de decisión reales. La familia de Tsallis revela que no estamos ante dos competidores rivales, sino ante puntos en un continuo matemático.

¿Cómo son de cercanas? Para una distribución binaria $(p, 1-p)$, podemos calcular ambas explícitamente:

$$\text{Gini}(p) = 2p(1-p)$$

$$\text{Entropía}(p) = -p \log_2 p - (1-p) \log_2(1-p)$$

Cuando $p = 0.5$, Gini = 0.5 y Entropía ≈ 1.0 (en bits). Ambas alcanzan su máximo. Cuando $p = 0.9$, Gini = 0.18 y Entropía ≈ 0.47. La escala es distinta, pero el comportamiento cualitativo es el mismo: crecen desde 0 hasta un máximo en $p = 0.5$, y decrecen simétricamente. En la práctica, normalizar uno por el otro produce árboles casi idénticos, porque el orden relativo de las ganancias es preservado.

## La ganancia de información como reducción de entropía condicional

Ahora vamos al corazón de CART: ¿qué hace que un split sea bueno? La respuesta es **información mutua**.

Supongamos que estamos en un nodo con muestras de diferentes clases. Podemos medir la incertidumbre sobre la clase usando la entropía de Shannon: $H(Y) = -\sum_k p_k \log p_k$. Ahora, consideramos un feature $X$ y evaluamos si dividir por él reduce esa incertidumbre.

Después del split en feature $X$ con umbral $t$, las muestras se distribuyen en dos subconjuntos: $X < t$ con proporción $p_L$ y $X \geq t$ con proporción $p_R$. Cada subconjunto tiene su propia entropía condicional: $H(Y|X < t)$ y $H(Y|X \geq t)$. La entropía condicional promedio es:

$$H(Y|X) = p_L H(Y|X < t) + p_R H(Y|X \geq t)$$

La **ganancia de información** es:

$$IG(Y, X) = H(Y) - H(Y|X)$$

Esto es exactamente la **información mutua** entre $X$ e $Y$:

$$I(X;Y) = H(Y) - H(Y|X)$$

¿Por qué es importante esta conexión? Porque la información mutua tiene un significado profundo en la Teoría de la Información de Shannon. Representa cuántos bits de incertidumbre sobre $Y$ son removidos si conocemos $X$. Equivalentemente, es cuántos bits de compresión ganamos si podemos usar $X$ para codificar $Y$ de forma más eficiente.

Considera un ejemplo. Si tenemos 100 muestras de dos clases, 50-50, la entropía es 1 bit. Para codificar cada etiqueta, necesitamos 1 bit. Ahora, descubrimos que si primero preguntamos por feature $X$, el 80% caen en una rama donde hay 90-10 en favor de una clase, y el 20% caen en otra con 10-90 inverso. Entonces $H(Y|X) \approx 0.47$ bits. La ganancia es $1 - 0.47 = 0.53$ bits. Esto significa que usando $X$ como "pregunta previa", reduce el promedio de bits necesarios para codificar $Y$ en un 53%.

Eso es exactamente lo que hace CART: busca el split que maximiza la ganancia de información, porque es el que más comprime la etiqueta. Este es un resultado bellísimo: **la división óptima en un árbol es aquella que maximiza la capacidad de codificación eficiente del label, en el sentido de Shannon.**

## ¿Por qué Gini es preferible en la práctica?

A pesar de que Entropía y Gini tienen fundamentos similares en la familia de Tsallis, en la práctica industrial Gini es dominante. ¿Por qué?

Primero, **razones computacionales**. Gini no tiene logaritmos. Calcular $1 - \sum p_k^2$ es operación entera en coma flotante directa, mientras que $-\sum p_k \log p_k$ requiere llamadas a funciones trascendentales. En un árbol con millones de posibles splits a evaluar, eso suma.

Segundo, **diferencias empíricas son mínimas**. Breiman et al. (1984) mostraron que para la mayoría de conjuntos de datos reales, los árboles entrenados con Gini versus Entropía son casi idénticos. Las diferencias en accuracy son típicamente menores al 1%.

Tercero—y aquí viene algo interesante—hay regiones donde difieren. Cuando las probabilidades de clase son muy extremas (cercanas a 0 o 1), la Entropía es más sensible. Considera una clase muy rara: $p_1 = 0.01, p_2 = 0.99$. Un split que mueve solo a la clase rara a una rama produce: $p_L = (0.5, 0.5), p_R = (0, 1)$ (imaginemos). Entropía en rama izquierda: 1 bit. Entropía en rama derecha: 0 bits. Gini en rama izquierda: 0.5. Gini en rama derecha: 0. La diferencia de sensibilidad es notable. En problemas con clases muy desbalanceadas, Entropía tiende a producir árboles más profundos que favorecen la detección de la clase rara. Esto puede ser deseable o no según el contexto.

## Limitaciones del criterio greedy de CART

Aquí llegamos al talón de Aquiles de CART: la **suboptimalidad local**. El algoritmo elige, en cada nodo, el split que maximiza localmente la ganancia de información. Pero esto no garantiza un árbol globalmente óptimo.

Un ejemplo clásico es el de los "Xor padre". Imagina dos features $X_1, X_2$ en $\{0, 1\}$ y label $Y = X_1 \oplus X_2$ (XOR). Hay dos splits posibles en cada rama: por $X_1$ o por $X_2$. CART mira el ganancia inmediata. Si elige $X_1$ en la raíz, genera dos subconjuntos casi mezclados. Luego, en la rama izquierda, $X_2$ produce una separación perfecta. En la rama derecha, $X_2$ también. El árbol resultante es óptimo, pero la elección no fue la más informativa localmente—fue equilibrada. Sin embargo, existen configuraciones más complejas donde elegir una feature "débil" en la raíz abre oportunidades para separaciones perfectas en niveles posteriores, mientras que elegir la feature "fuerte" localmente cierra esas oportunidades. Este es el problema de **lookahead**: saber a dos o tres niveles en el futuro cuál será la mejor estructura global.

Computacionalmente, garantizar la optimalidad global es intratable. El espacio de árboles posibles crece combinatoriamente con el número de features y muestras. En 2017, Bertsimas y Dunn propusieron métodos de **optimal decision trees** usando formulaciones de programación mixta-entera, pero están limitados a datasets pequeños. Para datos reales, CART greedy es el estándar porque es eficiente (O(n log n) para ordenar features) y produce resultados buenos en la práctica.

La lección es profunda: aceptamos la suboptimalidad local como el precio de la escalabilidad. Y paradójicamente, esa suboptimalidad es explotada por los ensembles: porque cada árbol greedy es ligeramente distinto (sensible a variaciones en los datos), cuando se agregan muchos, la varianza de esa suboptimalidad se reduce.

---

## Para reflexionar

1. La entropía de Tsallis interpola entre Gini ($q=2$) y Shannon ($q \to 1$) mediante el parámetro $q$. ¿Hay valores de $q$ distintos a 1 y 2 que hayan sido explorados sistemáticamente en la literatura de árboles de decisión? Si es así, ¿qué propiedades específicas del árbol mejoran o empeoran con esos valores intermedios?

2. La ganancia de información es equivalente a la información mutua $I(X;Y)$. En lugar de usar la ganancia discreta de CART (que asume que las probabilidades se estiman empíricamente de un sample finito), ¿qué pasaría si usaras una estimación no paramétrica de $I(X;Y)$ basada en k-vecinos más cercanos (por ejemplo, el método de Kraskov et al., 2004)? ¿Cuáles serían las ventajas y desventajas respecto a CART clásico?

3. La suboptimalidad global de CART es un problema NP-hard. Investigadores como Bertsimas & Dunn han propuesto formular la construcción de árboles como problemas de optimización exacta usando programación mixta-entera. ¿Cuáles son los principales obstáculos para escalar estos métodos exactos a datasets de millones de muestras? ¿Qué aproximaciones heurísticas o relaciones de relajación podrían explorar?

## Para ir más lejos

- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379–423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
- Tsallis, C. (1988). Possible generalization of Boltzmann-Gibbs statistics. *Journal of Statistical Physics*, 52(1–2), 479–487. https://doi.org/10.1007/BF01016429
- Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Chapman & Hall. Capítulos 4–5.
- Bertsimas, D., & Dunn, J. (2017). Optimal classification trees. *Machine Learning*, 106, 1039–1082. https://doi.org/10.1007/s10994-017-5633-9
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Capítulo 9.2.

---

*Lectura relacionada con la Clase ML_U2_C04 · Sección 2 · Assignment ML_A4*
