# La Sabiduría de las Multitudes en Machine Learning

**Unidad 2 · Lectura complementaria 13 · Audiencia: todos**

versión: 2025-1 | modificado: 2026-04-24 | lectura estimada: 5 min

---

En 1906, el estadístico victoriano Francis Galton asistió a una feria agrícola en Plymouth, Inglaterra, donde se realizaba un concurso peculiar: ochocientas personas apostaban al peso exacto de un buey. Galton, profundamente convencido de la superioridad del pensamiento experto sobre la intuición popular, sospechaba que el promedio de la multitud sería absurdo. Cuando finalmente calculó la mediana de las ocho mil apuestas (algunas personas participaban múltiples veces), quedó atónito: 1207 libras. El peso real del buey era 1198 libras. La multitud de granjeros aficionados había adivinado con un error de menos del uno por ciento. En ese momento, sin saberlo, Galton había descubierto un principio fundamental que siglos después sería central en el aprendizaje automático: la sabiduría de las multitudes.

## Las Condiciones del Milagro Estadístico

La anécdota de Galton no es magia estadística, sino consecuencia de condiciones precisas. Para que una multitud sea sabia, deben cumplirse tres requisitos ineludibles. Primero, la independencia: cada participante debe formar su estimación sin ser influenciado por el juicio de los otros. Si todos hubieran visto la apuesta del vecino antes de votar, la multitud habría simplemente amplificado un error inicial. Segundo, la diversidad: los participantes deben aportar perspectivas, información y métodos distintos. Un grupo de clones cometería idénticos errores. Tercero, la ausencia de sesgo sistemático compartido: no puede haber un error comprimido que todos cometan hacia la misma dirección. Si todos los participantes creyeran por alguna razón que el buey era más pesado de lo que era, la multitud erraría colectivamente.

Cuando estas tres condiciones se cumplen, algo extraordinario ocurre: los errores individuales se cancelan mutuamente. Algunos adivinos sobreestiman, otros subestiman; algunos se equivocan por marginalidad, otros son drásticamente errados. Pero en promedio, las desviaciones se anulan. El error del grupo es menor que el error promedio del individuo.

## Traducción al Lenguaje de los Ensambles

El salto desde la feria de Galton hacia los ensambles de modelos de aprendizaje automático es directo. Imagina que entrenas cinco modelos distintos para predecir si un cliente cancelará su suscripción a un servicio digital. Cada modelo aprende de los mismos datos, pero usa un algoritmo diferente: árbol de decisión, máquina de vectores de soporte, red neuronal, regresión logística, vecinos más cercanos. Ahora, en lugar de confiar en la predicción de un único modelo, combinas sus predicciones mediante promedio o votación. ¿Mejora el desempeño del ensemble respecto al modelo mejor? En la mayoría de los casos, sí.

La razón es precisamente la que Galton observó: si los cinco modelos cometen errores en direcciones distintas —si la red neuronal sobrestima la probabilidad de cancelación en ciertos perfiles de cliente mientras que el árbol la subestima en esos mismos perfiles— entonces el promedio de sus predicciones será más confiable que cualquiera de ellas por separado. Matemáticamente, si tenemos $n$ modelos cuyas predicciones tienen varianza $\sigma^2$ y sus errores tienen correlación $\rho$, entonces la varianza del promedio del ensemble es $\frac{\sigma^2}{n}(1 + (n-1)\rho)$. Si los modelos son independientes ($\rho = 0$), la varianza se reduce por un factor de $n$. Si son perfectamente correlacionados ($\rho = 1$), entrenar múltiples modelos no reduce varianza alguna.

Esta fórmula codifica una verdad incómoda: la diversidad es la moneda del ensemble. No puedes ganar precisión colectiva sin pagar el precio de la heterogeneidad individual. Dos modelos idénticos no forman un ensemble; son un modelo entrenado dos veces.

## Bagging versus Boosting: Dos Formas de Preguntar

Aunque tanto bagging como boosting son estrategias de ensamble inspiradas en la sabiduría de multitudes, operan bajo dinámicas radicalmente distintas. Bagging (del inglés bootstrap aggregating) responde a la pregunta: ¿qué pasa si hago la misma pregunta a múltiples expertos independientes que han visto datos ligeramente distintos? Cada modelo base se entrena con un subconjunto aleatorio de datos (muestreo con reemplazo, bootstrap). Sus errores son naturalmente independientes porque nacen de perspectivas desconectadas. El ensemble gana principalmente reduciendo varianza, porque los errores aleatorios de unos modelos cancelan los de otros.

Boosting, en cambio, implementa una consulta secuencial: le hago una pregunta al primer experto, veo dónde se equivoca, y le hago una pregunta más difícil al segundo experto que enfatiza exactamente esos errores. Así sucesivamente. Cada nuevo modelo aprende de los fracasos del anterior, ajustando los pesos de las muestras para concentrarse en las predicciones que eran erradas. Esta dinámica es más agresiva: en lugar de la independencia suave de bagging, boosting introduce una dependencia constructiva entre modelos. Su fortaleza es reducir el sesgo del modelo base, forzándolo a mejorar iterativamente.

Ambas estrategias son formas de sabiduría de multitudes, pero con distintos mecanismos. Bagging confía en la independencia; boosting confía en la retroalimentación y el aprendizaje colectivo.

## La Condición Crítica: Heterogeneidad de Errores

El punto de quiebre entre un ensemble que funciona y uno que falla reside en una pregunta simple: ¿equivocan en las mismas instancias? Si todos tus modelos cometen errores en exactamente las mismas muestras de datos, entonces combinarlos no aporta beneficio alguno. Imagina un dataset de imágenes de gatos y perros donde todos tus modelos fallan en distinguir un gato con un perro porque ambos están lado a lado. Entrenar cien modelos más no remediará este error compartido.

Por eso, en la práctica, la construcción de buenos ensambles es un arte de ingeniería consciente: se busca deliberadamente la diversidad de modelos base (algoritmos distintos, subconjuntos de features distintos, arquitecturas distintas). Es también por eso que bagging es tan efectivo con árboles de decisión: los árboles son modelos de alta varianza cuya topología cambia drásticamente con pequeños cambios en los datos. Entrenarlos con muestras bootstrap garantiza heterogeneidad natural de errores.

## Para reflexionar

1. En la democracia votamos para tomar decisiones colectivas, confiando en que muchas perspectivas distintas producen un resultado sabio. ¿Funcionan los ensambles de modelos por la misma razón que funciona la democracia? ¿En qué se diferencian fundamentalmente los supuestos sobre la naturaleza de los "votantes"?

2. Si todos tus modelos base fueron entrenados con exactamente los mismos datos, el mismo algoritmo y solo distintas semillas aleatorias del generador de números, ¿esperarías que su ensemble tenga un desempeño significativamente mejor? ¿Por qué sí o por qué no?

3. La fórmula $\frac{\sigma^2}{n}(1 + (n-1)\rho)$ sugiere que si $\rho = 1$ (correlación perfecta), la varianza del ensemble no se reduce. ¿Qué estrategias concretas de entrenamiento ayudarían a reducir la correlación entre modelos?

## Para ir más lejos

- Surowiecki, J. (2004). *The Wisdom of Crowds: Why the Many Are Smarter Than the Few and How Collective Wisdom Shapes Business, Economies, Societies and Nations*. Doubleday. ISBN: 978-0385503868.
- Krogh, A., & Vedelsby, J. (1995). Neural network ensembles, cross validation, and active learning. *Advances in Neural Information Processing Systems* (NIPS 1995), 231–237.
- Polikar, R. (2006). Ensemble based systems in decision making. *IEEE Circuits and Systems Magazine*, 6(3), 21–45. doi:10.1109/MCAS.2006.1688199
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. doi:10.1023/A:1010933404324

---

*Lectura relacionada con la Clase ML_U2_C05 · Secciones 1-4 · Lab ML_U2_Lab04 · Assignment ML_A5*
