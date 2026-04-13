# Dos filosofías para clasificar: modelos generativos vs discriminativos

**Unidad 2 · Lectura complementaria 5 · Audiencia: todos**
versión: 2025-1 | modificado: 2026-04-06 | lectura estimada: 9 min

---

En toda la historia del machine learning existe una tensión fundamental que a veces se olvida entre una práctica y otra: ¿debe un clasificador aprender **cómo es el mundo** o simplemente **dónde están los límites**?

Esta no es una pregunta trivial. La respuesta define dos familias completas de modelos con filosofías radicalmente distintas, fortalezas distintas y fallas distintas. Entender la diferencia no solo aclara por qué existen NB, LDA y QDA — también ilumina decisiones de diseño que seguirán apareciendo a lo largo de todo el curso.

---

## El detective y el árbitro

Imagina dos maneras de determinar si un correo es spam.

El **primer enfoque** estudia miles de correos spam y miles de correos legítimos. Aprende cómo son los correos de cada tipo: qué palabras usan, con qué frecuencia aparecen ciertas expresiones, cuál es la longitud típica. Cuando llega un correo nuevo, pregunta: ¿este correo se parece más a cómo son los spam o a cómo son los legítimos? Este enfoque construye un **modelo del mundo**: sabe generar correos plausibles de cada clase.

El **segundo enfoque** no aprende cómo son los correos. Solo aprende una regla de decisión: dado este conjunto de palabras, ¿a qué lado de una frontera cae? No sabe nada sobre cómo es un spam "típico". Solo sabe que si ciertas combinaciones aparecen, la predicción es spam.

El primer enfoque es un modelo **generativo**. El segundo, **discriminativo**.

---

## La diferencia técnica

Para un problema de clasificación con entrada $x$ y clase $y$, los dos enfoques modelan cantidades distintas:

Un modelo **generativo** modela la distribución conjunta $P(X, Y)$ o equivalentemente la verosimilitud $P(X \mid Y=k)$ junto con el prior $P(Y=k)$. A partir de estas piezas aplica el teorema de Bayes para obtener la probabilidad posterior $P(Y \mid X)$.

Un modelo **discriminativo** modela directamente $P(Y \mid X)$: aprende la frontera de decisión sin pasar por la distribución de los datos.

En la práctica esto se traduce en:

| | Generativo | Discriminativo |
|--|-----------|----------------|
| Qué modela | $P(X \mid Y)$ + $P(Y)$ | $P(Y \mid X)$ directamente |
| Ejemplos | Naive Bayes, LDA, QDA, GMM | Regresión Logística, SVM, redes neuronales |
| Puede generar muestras | Sí | No (solo clasifica) |
| Número de parámetros | Generalmente más | Generalmente menos |

---

## ¿Cuándo gana cada uno?

La respuesta clásica viene de un paper de Ng y Jordan (2002) que comparó Naive Bayes con Regresión Logística en decenas de problemas. Su hallazgo fue más matizado de lo esperado.

Con **pocos datos**, NB suele ganar. La razón es simple: NB asume más cosas sobre la distribución de los datos. Esos supuestos actúan como regularización implícita — acotan la búsqueda a modelos plausibles. Cuando los datos son escasos, tener más supuestos reduce la varianza de estimación.

Con **muchos datos**, la Regresión Logística suele alcanzar y superar a NB. Al relajar los supuestos sobre $P(X|Y)$, LR puede ajustarse mejor a la verdadera distribución cuando hay suficiente evidencia.

Hay un cruce: existe un número de muestras $n^*$ a partir del cual LR supera a NB. En los experimentos de Ng y Jordan, ese cruce ocurría típicamente entre 30 y 100 muestras dependiendo del problema.

```
   Error
    │
    │ \  NB
    │  \
    │   ──────────
    │       \   LR
    │        \──────────
    │
    └──────────────────────► n (muestras)
           n*
```

Este patrón explica por qué los modelos generativos siguen siendo relevantes en aplicaciones médicas, industriales y científicas donde los datos son costosos de recolectar.

---

## El caso de las distribuciones incorrectas

Hay una pregunta más sutil: ¿qué pasa cuando los supuestos del modelo generativo son falsos?

Naive Bayes asume que las features son condicionalmente independientes dado la clase. En el mundo real, esa independencia casi nunca se cumple exactamente. Y sin embargo, NB funciona sorprendentemente bien. ¿Por qué?

La respuesta tiene dos partes. Primero, para clasificar correctamente no necesitas estimar las probabilidades con precisión — solo necesitas que el orden relativo sea correcto. NB puede dar probabilidades mal calibradas y aun así clasificar bien. Segundo, incluso cuando las correlaciones son fuertes, a veces son iguales en todas las clases y se cancelan en la comparación.

Esto tiene un nombre: **Naive Bayes es un clasificador más robusto de lo que sus supuestos sugieren**. Es un ejemplo de que en ML práctica, los supuestos incorrectos no siempre implican malos resultados.

---

## Una consecuencia inesperada: detección de anomalías

Una ventaja exclusiva de los modelos generativos es que pueden hacer algo que los discriminativos no pueden: **evaluar qué tan probable es un punto bajo el modelo aprendido**.

Si un nuevo punto tiene $P(X=x)$ muy baja bajo todos los modelos de clase, el clasificador generativo puede detectarlo como una anomalía, incluso sin clase asignada. Un clasificador discriminativo como LR asignará una clase con alta confianza a cualquier punto, porque solo sabe dónde están las fronteras, no si el punto es plausible.

Esta diferencia es crucial en aplicaciones como detección de fraude o diagnóstico médico, donde los casos más peligrosos suelen ser los más raros y atípicos.

---

## Para reflexionar

1. Si quisieras construir un sistema que genere correos de spam realistas para entrenar un detector, ¿usarías un modelo generativo o discriminativo? ¿Por qué?

2. En diagnóstico oncológico, tienes 50 biopsias etiquetadas. ¿Empezarías con NB o Regresión Logística? ¿Cómo cambiaría tu elección si tuvieras 50.000?

3. ¿Se te ocurre un caso donde el supuesto incorrecto de un modelo generativo sea **beneficioso** (es decir, actúe como una buena regularización)?

---

## Para ir más lejos

- Ng, A. & Jordan, M. (2002). *On Discriminative vs. Generative Classifiers: A Comparison of Logistic Regression and Naive Bayes*. NIPS 2002.
- Bishop, C. (2006). *Pattern Recognition and Machine Learning*. Springer. Cap. 1.5.4.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Cap. 8.6.

---

*Lectura relacionada con la Clase ML_U2_C03 · Sección 1 (Marco Generativo)*
