# Cómo Leer e Interpretar un Árbol de Decisión

**Unidad 2 · Lectura complementaria 11 · Audiencia: Pregrado**

versión: 2025-1 | modificado: 2026-04-19 | lectura estimada: 11 min

---

¿Cuántos modelos de machine learning puedes explicarle a tu abuela en cinco minutos? Probablemente, solo uno: el árbol de decisión. "Abuela, si la persona gana menos de treinta mil pesos mensuales, le decimos no. Si gana más, pero tiene una deuda de más de cien mil, todavía le decimos no. De lo contrario, le decimos sí." Ella entiende. Las redes neuronales profundas, las máquinas de vectores soporte, los métodos de kernel—esos necesitarían una pizarra y una hora. Pero el árbol tiene esa virtud: es intuitivamente explicable.

Sin embargo, explicable no es sinónimo de fácil de leer. Un árbol de profundidad diez con mil nodos no se entiende de un vistazo, aunque en teoría sea "explicable". El desafío es desarrollar una literacia: saber dónde buscar, qué preguntas hacer, y cuándo el árbol te está contando la verdad o solo una aproximación conveniente de ella.

## El camino desde la raíz

Para predecir la clase de una nueva muestra usando un árbol, simplemente sigues un camino. Comienzas en la raíz—el nodo superior—y lees su pregunta. Por ejemplo: "¿el tamaño del tumor es menor a 20 milímetros?". Si la respuesta para tu muestra es sí, bajas por la rama izquierda. Si es no, bajas por la derecha. En el siguiente nodo, haces la misma cosa. Sigues bajando hasta que llegas a una hoja, que te da la predicción final.

Cada nodo te cuenta una historia. Junto a la pregunta, verás anotado cuántos datos de entrenamiento pasaron por ese nodo. Por ejemplo, podrías ver: "nodo raíz: 569 muestras, 357 benignas, 212 malignas". Eso te dice que de los 569 pacientes de entrenamiento, el algoritmo los dividió según el criterio, y la mayoría eran casos benignos. A medida que bajas por el árbol, estos números se hacen más pequeños (porque cada división partición el conjunto), pero la proporción de clases cambia. Idealmente, en los nodos más bajos, una clase domina completamente—o casi.

Veamos un árbol de diagnóstico ficticio pero verosímil:

```
                     tamaño_tumor < 20 mm?
                    /                      \
                   /                        \
              [569 casos]                [0 casos]
            benign: 357                      
            malign: 212                  
           /                        \        
    textura < 23?              forma_asimetría < 0.5?
    [400 casos]                 [169 casos]
    benign: 320                benign: 37
    malign: 80                 malign: 132
   /        \                  /         \
[LEAF]   [LEAF]           [LEAF]      [LEAF]
benign   malign           benign      malign
```

Cuando lees este árbol, desciendes por él como un mapa: si tamaño < 20, vamos al nodo izquierdo (textura < 23). Si tamaño >= 20, vamos al derecho (forma_asimetría < 0.5). En cada punto, ves cuántas muestras de entrenamiento alcanzaron eso—eso es importante, porque un nodo que solo tiene 5 muestras es más sospechoso que uno con 500. La predicción es más confiable cuando el nodo tiene muchas muestras de la clase ganadora.

## Leer la hoja: qué dice y qué no dice

La hoja es donde termina tu viaje. Supongamos que tu árbol llega a una hoja que dice: "malign | 132 muestras, 37 benignas, 132 malignas". ¿Qué significa exactamente?

Significa que de esos 169 pacientes de entrenamiento que satisficieron todas las condiciones de arriba (tamaño >= 20 AND forma_asimetría >= 0.5), 37 resultaron ser benignos y 132 malignos. El árbol, siendo conservador, predice la clase mayoritaria: **malign**. Esa es la predicción **dura**.

Pero hay incertidumbre. No es que estemos cien por ciento seguros. La fracción $\frac{132}{169} \approx 0.78$ nos dice la probabilidad empírica: si alguien cae en esa hoja, hay aproximadamente un 78% de chance de que sea maligno y un 22% de que sea benigno. Esta es la predicción **suave** o probabilidad. En scikit-learn, `predict()` te da la clase ganadora (la predicción dura), mientras que `predict_proba()` te da esas probabilidades.

Esa distinción es crucial cuando las decisiones tienen costos asimétricos. En diagnóstico médico, un falso negativo (decirle a alguien que está sano cuando está enfermo) suele ser más grave que un falso positivo (decirle que está enfermo cuando está sano). Entonces, podrías ajustar el umbral: en lugar de predecir malign cuando la probabilidad es > 0.5, podrías hacerlo cuando es > 0.3. Eso atraparías más casos verdaderos malignos, a costa de más alarmas falsas.

El árbol te proporciona los números crudos para tomar esa decisión, pero es **tu responsabilidad** hacer el ajuste. El algoritmo solo particionó el espacio; quién actúa según esa partición eres tú.

## Las features más importantes: qué significa el ranking

Después de entrenar un árbol, una métrica que surge naturalmente es: ¿cuáles features fueron más útiles para hacer las predicciones? Esta es la **importancia de features** (feature importance).

La idea es simple: si una feature fue usada para dividir muchos nodos, y esas divisiones redujeron significativamente la confusión (la impureza), entonces esa feature es importante. Formalmente, se mide cuánta "ganancia de información" (o "reducción de Gini") se logró usando esa feature.

Cuando ves un ranking de importancia que dice:

```python
# Importancia en el árbol entrenado
tamaño_tumor:      0.45
forma_asimetría:   0.30
textura:           0.15
color:             0.10
```

Significa que el tamaño del tumor fue responsable del 45% de las reducciones de impureza. Fue la herramienta más poderosa para separar benignos de malignos.

Pero aquí viene una advertencia crucial: **importancia no implica causalidad**. Si "tamaño del tumor" sale como la feature más importante, eso no significa que un tumor grande CAUSE malignidad. Podría ser que el tamaño sea un indicador observable de malignidad, pero la causalidad real esté en la biología subyacente (el tipo de célula, la presencia de mutaciones genéticas). O podría ser que el tamaño y la malignidad sean ambas consecuencias de un tercero no observado.

Para ilustrarlo, imagina un algoritmo entrenado en datos de un hospital donde los tumores malignos se detectan más tardío. Entonces, el tamaño será correlacionado con malignidad—los malignos son más grandes cuando se detectan. El árbol aprenderá "si es grande, probablemente sea maligno", lo que puede ser una buena predicción, pero no la razón causal. Confundir la correlación con la causalidad es una trampa común y peligrosa.

Veamos un pequeño código que ilustra esto:

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Datos ficticios
X = np.array([[20, 0.4], [25, 0.5], [15, 0.3], [30, 0.6]])
y = np.array([1, 1, 0, 1])  # 0=benigno, 1=maligno

tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X, y)

# Importancia
print(tree.feature_importances_)
# Output: [0.8, 0.2]  -> feature 0 (tamaño) es 4 veces más importante
```

En la práctica, cuando presentas un árbol, es responsabilidad tuya (o del dominio expert) interpretar esa importancia en contexto. ¿Tiene sentido que esa feature sea importante? ¿Hay mecanismos biológicos que lo expliquen? ¿O es posible que sea una correlación espuria?

## Cuándo el árbol es demasiado profundo para leerlo

Existe un dilema clásico en machine learning: **interpretabilidad versus rendimiento**. Un árbol de profundidad 3 es legible. Alguien puede seguir el camino y explicar su lógica. Pero probablemente tenga errores de entrenamiento relativamente altos—un accuracy del 85%, digamos.

Un árbol de profundidad 10 es ilegible. Tiene mil nodos. No podrías explicar a nadie por qué hizo una predicción específica sin un diagrama gigante y varias horas de trabajo. Pero tiene un accuracy del 95%.

¿Cuál eliges?

La respuesta depende del contexto. Si estás clasificando imágenes de gatos versus perros para un sistema de recomendación interna, elige el árbol profundo. Nadie necesita explicar la decisión. Si estás clasificando solicitudes de crédito, y regulaciones requieren que puedas explicar por qué rechazaste a alguien, necesitas el árbol simple, aunque sea menos preciso.

Una estrategia pragmática es usar el árbol profundo para predecir, pero el árbol simple para explicar. Entrenas dos modelos: uno grande para decisiones internas, uno pequeño para auditoría externa. Es un poco redundante, pero honesto.

En scikit-learn, el parámetro `max_depth` controla esto:

```python
tree_simple = DecisionTreeClassifier(max_depth=3)
tree_complex = DecisionTreeClassifier(max_depth=10)

tree_simple.fit(X_train, y_train)
tree_complex.fit(X_train, y_train)

# El tree_complex probablemente tenga mejor accuracy
# pero tree_simple es explicable
```

---

## Para reflexionar

1. Imagina que tuvieras que presentar el resultado de una decisión de préstamo a un comité de directivos no técnicos. ¿Usarías un árbol de profundidad 3 o uno de profundidad 10 para justificar por qué rechazaste o aprobaste a alguien? ¿Cómo manejarías la diferencia de accuracy entre ambos?

2. Una feature sale como la más importante en el árbol. ¿Cómo diseñarías un experimento o análisis para distinguir entre: (a) es verdaderamente un factor causal, (b) es una correlación con la causa verdadera, o (c) es una correlación espuria causada por un confundidor no observado?

3. Imagina que el árbol clasifica incorrectamente a un paciente: debería ser "maligno" pero predijo "benigno". ¿Cómo usarías la estructura del árbol (el camino que tomó, los nodos por los que pasó, sus números) para explicar qué salió mal y dónde falló el modelo?

## Para ir más lejos

- Molnar, C. (2022). *Interpretable Machine Learning*. 2nd ed. https://christophm.github.io/interpretable-ml-book/
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206–215. https://doi.org/10.1038/s42256-019-0048-x
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media. Capítulo 6.

---

*Lectura relacionada con la Clase ML_U2_C04 · Secciones 3 y 5 · Lab ML_U2_Lab03*
