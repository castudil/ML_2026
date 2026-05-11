# Backpropagation sin Miedo: Una Guía Visual e Intuitiva

**Unidad 3 · Lectura complementaria 02 · Audiencia: Pregrado**
versión: 2025-1 | modificado: 2026-05-09 | lectura estimada: 5 min

---

Pocos algoritmos en machine learning generan tanta ansiedad como backpropagation. El nombre suena técnico, la derivación completa involucra matrices y reglas de la cadena anidadas, y el resultado —que la red "aprenda"— parece casi mágico. Esta lectura no va a derivar nada. Su objetivo es construir una imagen mental tan clara que, cuando veas las fórmulas, te parezcan obvias.

Empieza por olvidar por un momento que se trata de redes neuronales. Imagina que tienes un proceso con varias etapas y quieres saber cómo cada etapa contribuye al resultado final.

## El Problema de la Culpa

Una red neuronal comete un error: predijo 0.3 cuando la respuesta correcta era 1. Ese error se mide con la función de pérdida —un número que resume qué tan equivocada está la red. Ahora tienes miles de pesos repartidos en varias capas. La pregunta central del entrenamiento es: ¿a cuánto del error contribuyó cada peso?

Si pudieras responder esa pregunta, sabrías en qué dirección mover cada peso para reducir el error. Y si reduces el error en miles de ejemplos, iterativamente, la red aprende.

El problema es que la relación entre un peso en la primera capa y el error final es muy indirecta. Ese peso afecta a una neurona, esa neurona afecta a la siguiente capa, esa capa afecta a la siguiente, y así hasta la salida. Es como preguntarle a una empresa qué gerente regional contribuyó más a un trimestre malo cuando hay cuatro niveles de gestión entre el gerente y el resultado financiero.

## La Regla de la Cadena: Una Herramienta Familiar

La regla de la cadena del cálculo dice algo simple: si $y$ depende de $u$ y $u$ depende de $x$, entonces cómo cambia $y$ cuando cambia $x$ es el producto de cómo cambia $y$ cuando cambia $u$ y cómo cambia $u$ cuando cambia $x$.

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

Backpropagation es exactamente esto, aplicado en cadena desde la salida hasta la entrada. El gradiente de la pérdida respecto a un peso en la primera capa se calcula multiplicando una serie de derivadas locales: de la pérdida respecto a la salida, de la salida respecto a la capa anterior, y así sucesivamente.

La razón por la que el algoritmo es *eficiente* es que calcula estas derivadas en orden reverso —de la salida hacia la entrada— y *reutiliza* los resultados intermedios. En lugar de calcular el camino completo para cada peso, calcula los factores compartidos una sola vez y los usa para todos los pesos de cada capa.

## Cómo Visualizarlo

Imagina la red como un grafo de flujo de izquierda a derecha: la entrada entra por la izquierda, fluye por las capas, y produce una predicción a la derecha. Durante el forward pass, calculas y guardas los valores intermedios —las activaciones de cada capa.

Durante el backward pass, haces el recorrido al revés. Comienzas con el error en la salida y lo "propagas hacia atrás", distribuyendo la responsabilidad a cada capa. Cada capa recibe un mensaje de "cuánto contribuiste al error" desde la capa siguiente, lo ajusta por la derivada de su propia función de activación, y lo pasa hacia la capa anterior.

Lo que fluye hacia atrás no son los datos originales —es información sobre el gradiente. Y es exactamente esa información la que te dice cómo ajustar los pesos.

## El Rol de las Activaciones No Lineales

Hay una razón específica por la que la derivada de la función de activación aparece en la fórmula de backpropagation: la activación es la parte no lineal de la transformación. Si la red usara solo transformaciones lineales, el gradiente fluiría sin cambios a través de todas las capas —y la red entera colapsaría en una sola transformación lineal.

La función ReLU, $f(z) = \max(0, z)$, tiene una derivada particularmente simple: es exactamente 1 cuando la entrada es positiva, y 0 cuando es negativa. Esto significa que el gradiente pasa intacto a través de las neuronas activas (las que tienen $z > 0$) y se bloquea completamente en las neuronas inactivas.

Esta propiedad tiene una consecuencia importante: si una neurona permanece inactiva para todos los ejemplos de entrenamiento —porque siempre recibe una entrada negativa— nunca recibirá gradiente y sus pesos nunca se actualizarán. La neurona está "muerta". Este es el problema conocido como *dying ReLU*, y es una de las razones por las que la inicialización de los pesos importa más de lo que parece.

## Una Intuición Final: Descenso por el Terreno

Piensa en la función de pérdida como un paisaje topográfico. Cada combinación posible de pesos de la red define un punto en ese paisaje, con una altitud proporcional al error que comete. El objetivo del entrenamiento es encontrar el punto más bajo —el valle que representa el menor error posible.

Backpropagation te da la pendiente del terreno en el punto donde estás. El gradiente descendente te dice: muévete en la dirección opuesta a la pendiente. Si estás en una colina, el gradiente apunta cuesta arriba; moverse en sentido contrario te lleva cuesta abajo.

Lo que hace que el entrenamiento sea difícil no es el cálculo del gradiente —backpropagation lo resuelve— sino la geometría del terreno: tiene miles de millones de dimensiones, puede tener muchos valles locales, y la pendiente puede ser casi plana en algunas regiones (el problema del gradiente que desaparece). Entender backpropagation es solo el primer paso; entender el terreno que navega es la pregunta que ocupa a los investigadores hasta hoy.

## Para reflexionar

1. La derivada de ReLU es 0 para entradas negativas. ¿Qué implica esto para el aprendizaje de una neurona que siempre recibe entradas negativas? ¿Cómo afectaría esto al entrenamiento de una red si muchas neuronas están en esa situación?

2. Backpropagation calcula los gradientes exactos. ¿Significa eso que el modelo llegará al mínimo global de la función de pérdida? ¿Por qué sí o por qué no?

3. En el forward pass, guardamos los valores intermedios de activación de cada capa. ¿Por qué es necesario guardarlos? ¿Qué pasaría si no los guardáramos y tuviéramos que recalcularlos durante el backward pass?

## Para ir más lejos

- Nielsen, M. (2015). *Neural Networks and Deep Learning*. Cap. 2: How the backpropagation algorithm works. [Online, gratuito] http://neuralnetworksanddeeplearning.com/chap2.html

- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536. doi:10.1038/323533a0

- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3ª ed.). O'Reilly. Cap. 10, sección "Training Deep Neural Networks".

- Karpathy, A. (2022). *micrograd*: motor de autodiferenciación en ~100 líneas de Python. https://github.com/karpathy/micrograd [Implementación minimalista, excelente para entender el algoritmo]

---
*Lectura relacionada con la Clase ML_U3_C01 · Sección 4 (Backpropagation)*
*· Lab ML_U3_Lab01*
