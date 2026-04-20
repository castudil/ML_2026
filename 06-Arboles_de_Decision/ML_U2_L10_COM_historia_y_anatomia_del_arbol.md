# Historia y Anatomía de los Árboles de Decisión

**Unidad 2 · Lectura complementaria 10 · Audiencia: todos**

versión: 2025-1 | modificado: 2026-04-19 | lectura estimada: 12 min

---

Es paradójico que el método que parece más humano en machine learning—una serie de preguntas de sí o no que excluyendo opciones nos lleva a una conclusión—haya tardado décadas en encontrar un fundamento matemático riguroso. Los médicos diagnosticaban por eliminación desde hace siglos. Los árbitros de ajedrez descartaban movimientos ilegales en fracciones de segundo. Pero no fue hasta los años sesenta del siglo veinte que Hunt, Marin y Stone se preguntaron: ¿podemos formalizar este proceso? ¿Podemos enseñar a una máquina a construir sus propios árboles de preguntas a partir de datos?

La historia de los árboles de decisión no es la historia de una idea revolucionaria, sino de una lenta formalización de lo obvio. Y esa lentitud es instructiva.

## El árbol antes del árbol

Cuando Edward Feigenbaum y Joshua Lederberg querían que sus computadoras diagnosticaran infecciones bacterianas en los años sesenta, no pensaban en árboles. Pensaban en reglas: "si el paciente tiene fiebre y los cultivos muestran cocos grampositivos, entonces…" Pero Hunt y sus colegas vieron en esos sistemas de reglas encadenadas un patrón geométrico: la partición recursiva del espacio de features. El trabajo de 1962 (Concept Learning System, CLS) fue el primero en automatizar la construcción de esas particiones a partir de ejemplos etiquetados. Sin embargo, el CLS trabajaba solo con atributos categóricos y requería exhaustividad en los datos—un lujo raro en el mundo real.

La verdadera revolución vino con J. Ross Quinlan. En 1986, Quinlan introdujo ID3 (Iterative Dichotomiser 3), un algoritmo que abrazaba completamente la Teoría de la Información de Shannon. La idea era simple pero potente: en cada nodo, elige la pregunta (el split de features) que maximiza la ganancia de información, es decir, que reduce más la entropía de las etiquetas. ID3 fue elegante, rápido, y funcionaba en problemas reales. Pero tenía limitaciones estructurales: solo aceptaba atributos categóricos, y su árbol no era binario—podía crear múltiples ramas desde un nodo. Eso hacía que los árboles crecieran de forma descontrolada, especialmente cuando había atributos con muchas categorías.

Mientras Quinlan refinaba ID3 hacia C4.5, en California, Leo Breiman, Jerome Friedman, Richard Olshen y Charles Stone publicaban en 1984 *Classification and Regression Trees* (CART). CART cambió dos cosas fundamentales. Primero, usaba el índice de Gini en lugar de entropía—una métrica más computacionalmente eficiente que producía árboles similares en la práctica. Segundo, insistía en que el árbol fuera binario: cada nodo se divide exactamente en dos ramas. Esto parecía una restricción, pero resultó ser una bendición. Los árboles binarios son más pequeños, más fáciles de podar, más estables. Y CART no discriminaba entre atributos categóricos y continuos: ambos se trataban como umbrales de corte. Un atributo continuo se dividía en $x < t$ o $x \geq t$ para algún umbral $t$. Un atributo categórico se dividía en $x \in S$ o $x \notin S$ para algún subconjunto $S$ de categorías.

Con el tiempo, CART se convirtió en el estándar. No porque fuera perfecto—Quinlan y otros siguieron mejorando ID3/C4.5—sino porque era robusto, práctico y se generalizaba bien. Hoy, librerías como scikit-learn y R's rpart usan variaciones de CART por defecto.

## Anatomía de un árbol

Imaginemos un árbol que clasifica flores del iris: si el largo del sépalo es menor que 5.9 cm, vamos al nodo izquierdo; si no, al derecho. En el nodo izquierdo, preguntamos por el ancho del sépalo. Y así sucesivamente, hasta que llegamos a una hoja—un nodo terminal que no se divide más.

```
                   largo_sépalo < 5.9?
                  /                    \
              SÍ /                      \ NO
               /                        \
            [99 muestras]          [51 muestras]
           ancho_sépalo < 3.1?    largo_pétalo < 4.9?
           /              \         /              \
        ...              ...      ...            ...
        
        [HOJA]          [HOJA]   [HOJA]        [HOJA]
       setosa         versicolor virginica     virginica
```

La anatomía tiene nombres precisos. El **nodo raíz** es aquel donde comienza el árbol, arriba del todo. Los **nodos internos** son aquellos que se dividen en dos hijos. Las **hojas** (o nodos terminales) son aquellos que no se dividen—dan una predicción. Cada nodo interno posee un **criterio de división**: una pregunta de la forma "¿feature $X$ es menor que umbral $t$?". La **profundidad** de un nodo es cuántos pasos hay desde la raíz hasta él. La **altura** del árbol es la profundidad máxima de cualquier hoja.

Lo crucial es lo que sucede geométricamente. Si el espacio de features es bidimensional (dos features continuas), cada división crea un rectángulo nuevo. La primera división en $X_1 < t_1$ divide el plano verticalmente. Luego, cada subregión se divide nuevamente, por ejemplo en $X_2 < t_2$, creando más rectángulos. El árbol final particiona el espacio bidimensional en un conjunto de rectángulos alineados con los ejes. En tres dimensiones, obtendríamos cubos alineados. En $d$ dimensiones, hipercubos. Por eso decimos que un árbol de decisión es un aproximador de funciones que usa fronteras paralelas a los ejes.

En cada nodo se almacena información valiosa: el número de muestras de entrenamiento que lo alcanzaron, cuántas son de cada clase, y una métrica de "impureza" que mide cuán mezcladas están esas clases. Si un nodo tiene 100 muestras y todas son de la clase A, su impureza es cero—es puro. Si tiene 50 de clase A y 50 de clase B, su impureza es máxima.

## ¿Por qué funciona tan bien y tan mal?

Los árboles de decisión tienen una virtud que pocos modelos poseen: capturan **interacciones** entre features de forma natural. Consideremos un problema clásico: clasificar si alguien pagará un préstamo. Una regresión logística solo puede capturar efectos lineales: "si el ingreso es alto, probabilidad de pago aumenta". Pero los árboles pueden capturar: "si el ingreso es alto Y el historial crediticio es malo, entonces no pagues". Eso es una interacción de segundo orden, y el árbol la ve automáticamente porque a cada lado de una división, vuelve a evaluar las otras features. Los bosques de árboles (que veremos en la próxima unidad) aprovechan precisamente esta habilidad.

Pero hay un lado oscuro. Los árboles de decisión son **inherentemente inestables**. Imagina que tus datos tienen dos features que son casi idénticas en su poder predictivo. Un pequeño cambio en el conjunto de entrenamiento—quizá una o dos muestras—puede hacer que el árbol elija una feature en lugar de la otra en la raíz, produciendo un árbol completamente diferente con las mismas predicciones finales pero una estructura distinta. Cambio la raíz, y todo lo que cuelga de ella se invierte. Esa inestabilidad es exasperante si quieres un modelo interpretable y confiable.

Sin embargo, esa misma inestabilidad es una bendición disfrazada. Porque si pequeños cambios en los datos crean árboles diferentes, entonces crear muchos árboles con muestras ligeramente diferentes—bagging, boosting—produce modelos heterogéneos cuyas predicciones se promedian para ser mucho más estables y precisas que un árbol solo. Eso es la filosofía de los ensembles, que es el tema de la próxima unidad. La inestabilidad del árbol individual es exactamente lo que lo hace valioso en agregación.

---

## Para reflexionar

1. El médico que diagnostica por eliminación tiene décadas de experiencia codificadas en su árbol mental: sabe que ciertos síntomas aparecen juntos, que el contexto importa, que hay excepciones a las reglas. ¿Cómo se compara eso con lo que aprende un algoritmo como CART de un dataset finito? ¿Qué ventajas y desventajas tiene cada uno?

2. Si el espacio de features es un hipercubo de alta dimensión y un árbol lo divide en rectángulos alineados con los ejes, ¿qué formas de frontera de decisión quedan por completo fuera de su alcance? ¿Puedes pensar en un tipo de patrón que los árboles nunca podrían capturar bien?

3. La inestabilidad del árbol—que un conjunto de datos ligeramente distinto produce una estructura completamente diferente—¿es fundamentalmente un defecto del modelo o una propiedad que es aprovechable? ¿Cómo lo relacionarías con la idea de diversity (diversidad) en los ensembles que estudiaremos?

## Para ir más lejos

- Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning*, 1(1), 81–106. https://doi.org/10.1007/BF00116251
- Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Chapman & Hall.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media. Capítulo 6.

---

*Lectura relacionada con la Clase ML_U2_C04 · Secciones 1 y 3 · Lab ML_U2_Lab03 · Assignment ML_A4*
