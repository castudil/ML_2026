# La geometría de las fronteras de decisión

**Unidad 2 · Lectura complementaria 7 · Audiencia: Pregrado**
versión: 2025-1 | modificado: 2026-04-06 | lectura estimada: 9 min

---

Cuando un clasificador aprende, lo que realmente hace es **dividir el espacio de features en regiones**. Cada región corresponde a una clase. La línea (o curva, o superficie) que separa esas regiones es la **frontera de decisión**, y su forma revela más sobre el modelo que cualquier número de métricas.

Esta lectura es una guía visual para entender qué forma toma esa frontera en cada uno de los tres clasificadores gaussianos: Gaussian NB, LDA y QDA. No hay derivaciones — solo geometría e intuición.

---

## El espacio de features como territorio

Imagina que tus datos son puntos en un mapa. Tienes dos features: el eje horizontal y el vertical. Cada punto tiene un color según su clase. El clasificador traza líneas en ese mapa para asignar cada región a una clase.

Lo primero que hay que internalizar es que **la frontera de decisión no es un objeto que el modelo dibuja explícitamente**. Es la consecuencia de cómo el modelo asigna probabilidades: la frontera es donde dos clases tienen igual probabilidad. A ambos lados de esa línea, una clase domina.

El tipo de frontera que emerge depende directamente de los supuestos del modelo.

---

## Gaussian NB: fronteras que respetan los ejes

GNB asume que cada feature es independiente dada la clase. Esto equivale a asumir que las distribuciones de cada clase son **elipses alineadas con los ejes** en 2D (elipsoides en más dimensiones). No hay inclinación: los ejes de la elipse coinciden exactamente con los ejes de features.

```
feature 2
    ↑
    │    ●●●       ▲▲▲
    │   ●●●●●     ▲▲▲▲▲
    │    ●●●       ▲▲▲
    │         |
    └──────────────────→ feature 1
              ↑
         frontera de NB
         (línea recta aquí porque las varianzas son iguales)
```

Si las varianzas de las features son iguales en ambas clases, la frontera es una **línea recta**. Si son distintas, se curva — pero siempre respetando la orientación de los ejes. Nunca produce una frontera diagonal en el sentido general, porque NB ignora las correlaciones.

**Consecuencia práctica:** si tus clases están separadas por una frontera diagonal (por ejemplo, los puntos de la clase A tienden a tener feature1 alta cuando feature2 es baja), NB no puede capturar esa diagonal. LDA sí.

---

## LDA: la frontera más simple que puede existir

LDA asume que todas las clases tienen la **misma forma**, solo desplazadas. Matemáticamente, comparte una sola matriz de covarianza entre todas las clases. Como las formas son iguales, al comparar dos clases todo lo que importa es la diferencia de centros. La frontera resultante es **un hiperplano** (una línea en 2D).

```
feature 2
    ↑
    │  ●●           ▲▲
    │ ●●●●  /      ▲▲▲▲
    │  ●●  /        ▲▲
    │     /
    └────/──────────────→ feature 1
        ↑
    frontera LDA
    (línea, posiblemente diagonal)
```

La dirección de esa línea es perpendicular al vector que une los centros de las clases, pero ajustada por la forma compartida (la covarianza). Si la covarianza tiene correlaciones, la frontera se inclina. Esto es exactamente lo que NB no puede hacer.

Una propiedad útil de LDA: con $K$ clases produce $K-1$ ejes discriminantes. Para $K=3$ clases (como en Iris o Wine), proyecta a un plano 2D, lo que permite **visualizar la separación en 2D sin importar cuántas features tengan los datos originalmente**.

---

## QDA: cuando cada clase tiene su propia forma

QDA relaja el supuesto de covarianza compartida. Ahora cada clase tiene su propia elipse, con su propia orientación y tamaño. La frontera entre dos clases ya no puede ser una línea — tiene que ser una curva que siga las geometrías distintas de cada elipse.

```
feature 2
    ↑
    │  ●●●         ▲▲▲▲
    │ ●●●●●   (   ▲▲▲▲▲▲
    │  ●●●     )   ▲▲▲▲
    │           )
    └───────────────────→ feature 1
               ↑↑
         frontera QDA
         (curva, puede ser parábola,
          elipse o hipérbola)
```

La forma exacta de la curva depende de las covarianzas. Puede ser una parábola, una elipse, una hipérbola, o incluso dos ramas separadas. Esta flexibilidad le da a QDA mayor capacidad expresiva, pero también más riesgo de sobreajuste.

---

## La tabla que lo resume todo

| Modelo | Forma de la elipse por clase | Frontera resultante |
|--------|------------------------------|---------------------|
| GNB | Ejes alineados (diagonal = 0) | Cuadrática, ejes alineados |
| LDA | Igual para todas las clases | **Línea recta** (hiperplano) |
| QDA | Distinta por clase | **Curva** (cónica general) |

---

## Cómo leer una frontera de decisión en la práctica

Cuando visualizas una frontera de decisión, hay tres preguntas útiles:

¿Es suave o tiene bordes? Una frontera con muchos quiebres sugiere un modelo complejo que puede estar sobreajustando. Las fronteras de NB, LDA y QDA son siempre suaves (analíticas), lo que es una ventaja.

¿Hay regiones "islas" dentro del territorio de otra clase? Si una clase tiene una región desconectada, puede ser una señal de ruido en los datos o de un modelo que memorizó anomalías. Los modelos gaussianos no producen islas a menos que los datos las justifiquen fuertemente.

¿La frontera tiene sentido para el problema? Si clasificas tumores y la frontera de decisión dice que un tumor puede ser maligno con alta confianza solo porque una feature tiene un valor extremo, vale la pena preguntarse si esa feature realmente debería tener ese peso.

---

## Para reflexionar

1. Dibuja a mano un dataset 2D con dos clases donde LDA funcione perfectamente y NB falle. ¿Qué condición deben cumplir los datos para que eso ocurra?

2. Ahora dibuja un dataset donde QDA funcione mejor que LDA. ¿Qué diferencia hay entre los dos casos?

3. ¿Podrías construir un dataset donde los tres modelos tengan exactamente la misma frontera de decisión? ¿Qué propiedades tendrían que tener los datos?

---

## Para ir más lejos

- Géron, A. (2022). *Hands-On Machine Learning* (3rd ed.). O'Reilly. Cap. 2 y 4.
- James et al. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. Cap. 4.4. Disponible en: statlearning.com
- Hastie, Tibshirani & Friedman (2009). *The Elements of Statistical Learning*. Springer. Cap. 4.3.

---

*Lectura relacionada con la Clase ML_U2_C03 · Sección 5 (Experimento comparativo: fronteras de decisión)*
