# El Cerebro que Aprende: Historia y Filosofía de las Redes Neuronales

**Unidad 3 · Lectura complementaria 01 · Audiencia: todos**
versión: 2025-1 | modificado: 2026-05-09 | lectura estimada: 6 min

---

En 1943, un neurofisiólogo y un matemático se sentaron a imaginar una máquina que pensara. Warren McCulloch y Walter Pitts propusieron que una neurona podía modelarse como una compuerta lógica: recibe señales, las suma, y dispara si la suma supera un umbral. Era una idea extraña para la época —traducir la biología en álgebra— y resultó ser una de las semillas más fértiles de la historia de la computación.

Lo que siguió no fue un progreso lineal. Fue una historia de promesas exageradas, silencios dolorosos y resurrecciones inesperadas. Entender esa historia no es un ejercicio de nostalgia: es esencial para saber por qué las redes neuronales funcionan como funcionan, y para no repetir los errores de quienes prometieron demasiado demasiado pronto.

## El Primer Verano: Rosenblatt y el Perceptrón

Frank Rosenblatt era un psicólogo con la energía de un evangelista. En 1958, presentó el perceptrón como una máquina capaz de aprender a reconocer patrones visuales. El New York Times escribió que la Armada esperaba una máquina que "podrá caminar, hablar, ver, escribir, reproducirse y ser consciente de su existencia". Era 1958, y la exageración ya era parte del negocio.

El perceptrón tenía un poder real: dado un conjunto de datos linealmente separables, convergía. El teorema de convergencia, demostrado por Novikoff, garantizaba que el algoritmo de Rosenblatt encontraría una solución si existía. Pero esa garantía tenía una trampa escondida: el "si existía".

En 1969, Marvin Minsky y Seymour Papert publicaron *Perceptrons*, un libro que demostró con precisión matemática que el perceptrón simple no podía aprender XOR, ni detectar la conectividad de una figura, ni resolver una docena de problemas aparentemente simples. No era una crítica superficial. Era una disección quirúrgica. El campo entró en su primer invierno.

## El Invierno y el Deshielo

Durante los años 70, el financiamiento para redes neuronales se evaporó. Los "sistemas expertos" —programas basados en reglas escritas por humanos— dominaban la inteligencia artificial. La intuición era seductora: si queremos máquinas inteligentes, enseñémosles las reglas explícitamente.

El problema de los sistemas expertos no era intelectual sino práctico. Codificar el conocimiento de un médico, un abogado o un controlador de tráfico en reglas explícitas resultó ser una tarea monumental e interminable. El conocimiento humano, se descubrió dolorosamente, no vive principalmente en reglas —vive en patrones, en ejemplos, en intuiciones difíciles de articular.

La respuesta llegó en 1986, desde un paper cuyo título es casi anticlimático: "Learning representations by back-propagating errors". Rumelhart, Hinton y Williams demostraron que el gradiente de la pérdida podía calcularse eficientemente en redes de múltiples capas usando la regla de la cadena. El algoritmo no era nuevo en teoría —Werbos lo había derivado en su tesis de 1974— pero la implementación práctica y la demostración empírica en problemas reales abrieron el segundo verano.

## Por Qué Importa la Profundidad

El argumento técnico a favor de las redes profundas no es intuitivo. El Teorema de Aproximación Universal dice que una sola capa oculta suficientemente grande puede aproximar cualquier función continua. Entonces, ¿para qué añadir capas?

La respuesta está en la *eficiencia representacional*. Ciertas funciones que requieren un número exponencial de neuronas en una red poco profunda pueden representarse con un número polinomial en una red profunda. La composición de funciones simples —lo que hace cada capa— permite capturar jerarquías de abstracción que son inherentes en muchos datos naturales: en una imagen, los píxeles forman bordes, los bordes forman texturas, las texturas forman partes, las partes forman objetos. Ninguna capa lo ve todo; cada capa ve un nivel de la jerarquía.

Esta no es solo una ventaja computacional. Es una correspondencia profunda entre la estructura de los datos del mundo y la arquitectura de las redes que los modelan.

## El Tercer Verano y Sus Advertencias

En 2012, una red neuronal llamada AlexNet ganó la competencia ImageNet reduciendo el error en un 40% de golpe. La comunidad reaccionó con algo parecido al asombro. Siguió una década de avances extraordinarios: traducción automática, síntesis de voz, diagnóstico médico, generación de texto.

Pero la historia enseña precaución. Cada verano ha ido seguido de un invierno, y el patrón no es accidental: las expectativas tienden a superar la realidad. Hoy las redes neuronales profundas tienen limitaciones bien documentadas —necesitan grandes cantidades de datos, son difíciles de interpretar, pueden fallar catastróficamente en situaciones no vistas, y consumen recursos energéticos enormes. Ninguna de estas limitaciones es insuperable, pero ninguna ha desaparecido.

Estudiar redes neuronales en 2025 significa estudiar una tecnología poderosa y madura, pero también una con tensiones no resueltas. ¿Cuándo hay que usarlas y cuándo preferir un modelo más simple y transparente? ¿Cómo garantizar que el comportamiento en producción coincide con el comportamiento en evaluación? Estas preguntas no tienen respuestas técnicas únicas —requieren juicio, contexto y, a veces, honestidad sobre lo que no sabemos.

## Para reflexionar

1. El perceptrón fue propuesto en 1958 y el MLP con backpropagation no llegó hasta 1986. ¿Qué le impidió al campo avanzar antes? ¿Fue una limitación técnica, computacional, o algo más profundo en cómo la comunidad científica evalúa las ideas?

2. Los sistemas expertos de los años 70 intentaban codificar conocimiento como reglas explícitas. Las redes neuronales lo aprenden de datos. ¿Qué problemas resuelve mejor cada enfoque? ¿En qué dominio de tu carrera aplicarías cada uno?

3. Cada "invierno de la IA" fue precedido por promesas excesivas. ¿Ves patrones similares en la retórica actual sobre inteligencia artificial generativa? ¿Qué criterios usarías para distinguir avance genuino de hipérbole?

## Para ir más lejos

- McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *Bulletin of Mathematical Biophysics*, 5(4), 115–133. doi:10.1007/BF02478259

- Rosenblatt, F. (1958). The Perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386–408. doi:10.1037/h0042519

- Minsky, M., & Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry*. MIT Press. [Clásico sobre las limitaciones del perceptrón simple]

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536. doi:10.1038/323533a0

- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521, 436–444. doi:10.1038/nature14539 [Revisión accesible del estado del arte]

---
*Lectura relacionada con la Clase ML_U3_C01 · Sección 1 (Motivación e Historia)*
*· Assignment ML_A7*
