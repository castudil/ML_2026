# K-means: La Geometría de los Promedios

**Unidad 4 · Lectura complementaria 02 · Audiencia: pregrado**
versión: 2025-1 | modificado: 2026-05-30 | lectura estimada: 6 min

---

Hay una idea elegante en el corazón de K-means: los grupos naturales en los datos son aquellos donde cada punto está más cerca de su propio centro que de cualquier otro centro. Esta observación, simple como parece, genera todo el mecanismo del algoritmo.

## La Intuición de los Supermercados

Considera el siguiente problema: una cadena de supermercados quiere abrir tres nuevas tiendas en una ciudad. Quiere minimizar la distancia promedio que camina cada habitante hasta su supermercado más cercano. ¿Dónde ubica las tres tiendas?

La respuesta no es obvia, pero hay un procedimiento iterativo que la aproxima bien:

Primero, escoge al azar tres ubicaciones iniciales para las tiendas. Segundo, asigna a cada habitante el supermercado más cercano —esto divide la ciudad en tres zonas. Tercero, muda cada supermercado al centro geográfico de los habitantes que tiene asignados. Cuarto, repite desde el segundo paso.

En cada iteración, la distancia total recorrida por todos los habitantes solo puede bajar o mantenerse igual —nunca sube. El proceso converge cuando nadie cambia de supermercado entre una iteración y la siguiente.

Eso es exactamente K-means. Los "supermercados" son los centroides. Los "habitantes" son los puntos de datos. La "distancia total" es la inercia o WCSS.

## Dos Pasos que se Alternan

El algoritmo K-means alterna entre dos operaciones:

**Paso de asignación (Paso E):** cada punto se asigna al centroide más cercano según la distancia euclidiana. Si los centroides son $\mu_1, \mu_2, \ldots, \mu_K$, el punto $x_i$ se asigna al cluster $k^* = \arg\min_k \|x_i - \mu_k\|^2$.

**Paso de actualización (Paso M):** cada centroide se recalcula como la media aritmética de todos los puntos asignados a ese cluster: $\mu_k^{\text{nuevo}} = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$.

¿Por qué la media? Porque la media minimiza la suma de distancias cuadráticas: si quieres un único punto que esté lo más cerca posible de todos los puntos de tu cluster (en el sentido de minimizar la suma de cuadrados), ese punto es la media. Cualquier otro punto daría una inercia mayor.

Esta propiedad matemática garantiza que el Paso M nunca puede aumentar la inercia. Y el Paso E tampoco puede aumentarla —asignar un punto a un centroide diferente solo ocurre si ese centroide está más cerca. La combinación de dos pasos que solo bajan o mantienen la inercia garantiza que el algoritmo converge.

## El Problema de los Mínimos Locales

La convergencia es real, pero tiene una trampa: K-means converge a un *mínimo local*, no necesariamente al *mínimo global*. La inercia mínima que encuentras depende de dónde empezaron los centroides.

Una inicialización desafortunada puede llevar a situaciones como esta: todos los centroides iniciales están agrupados en una sola región del espacio. Los clusters resultantes cortan el espacio de maneras subóptimas que ninguna cantidad de iteraciones puede corregir, porque localmente todo está en equilibrio.

La solución más simple es correr el algoritmo muchas veces con inicializaciones diferentes y quedarse con la solución de menor inercia. En scikit-learn, el parámetro `n_init` controla esto —el default es 10 corridas.

La solución más elegante es K-means++, una forma inteligente de elegir los centroides iniciales que los distribuye bien en el espacio. La idea: el primer centroide se elige al azar entre los puntos. Cada centroide siguiente se elige con probabilidad proporcional al cuadrado de la distancia al centroide más cercano ya elegido —los puntos más alejados del "territorio" ya cubierto tienen más probabilidad de ser elegidos. El resultado es una distribución inicial que ya tiene cierta separación entre centroides, lo que típicamente lleva a mejores soluciones finales.

## El Diagrama de Voronoi

Hay una forma bonita de visualizar qué hace K-means. Dados K centroides en el espacio, el *diagrama de Voronoi* divide el espacio en K regiones: cada región contiene todos los puntos más cercanos a un centroide particular que a cualquier otro.

La frontera entre dos regiones es el bisector perpendicular del segmento que conecta los dos centroides. En 2D, estas fronteras son segmentos de recta. En dimensiones más altas, son hiperplanos.

K-means, en esencia, busca los K centroides que definen el mejor diagrama de Voronoi posible: aquel donde los puntos dentro de cada región están lo más cerca posible de su centroide. La frontera entre clusters no es nunca una curva arbitraria —siempre es un hiperplano. Esto es una consecuencia directa de usar la distancia euclidiana y es precisamente lo que hace que K-means falle en clusters con formas no convexas.

## Cuándo Funciona y Cuándo No

K-means funciona bien cuando los clusters son aproximadamente esféricos, tienen tamaño similar y están bien separados. El dataset de Iris ilustra tanto su capacidad (recupera bien el cluster de *Iris setosa*, que está muy separado) como sus limitaciones (confunde *Iris versicolor* e *Iris virginica*, que se solapan).

K-means falla sistemáticamente cuando:
- Los clusters tienen formas no convexas (lunas, anillos) — las fronteras de Voronoi no pueden capturar esa geometría
- Los clusters tienen densidades o tamaños muy distintos — K-means tiende a dividir los clusters grandes para minimizar la inercia
- Hay outliers significativos — la media es sensible a valores extremos

Para estos casos existen alternativas: DBSCAN para clusters de densidad variable y formas arbitrarias, GMM para clusters elípticos, clustering espectral para estructuras no lineales.

## Para reflexionar

1. Si K-means minimiza la inercia (WCSS), y la inercia siempre baja durante el entrenamiento, ¿por qué un K más grande no es siempre mejor? ¿Cuál es la inercia mínima posible, y qué K la alcanza?

2. Supón que un cluster tiene un solo punto muy alejado del centroide. ¿Cómo afecta ese punto al centroide en la siguiente iteración? ¿Qué pasaría si lo removieras del dataset?

3. K-means usa la media como centroide. Si tus datos son latitudes y longitudes en la superficie terrestre, ¿habría algún problema con usar la media euclidiana como centroide? ¿Qué alternativa propones?

## Para ir más lejos

- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129-137. [El paper original de K-means]

- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *Proceedings of SODA 2007*. [K-means++]

- Géron, A. (2022). *Hands-On Machine Learning* (3ª ed.). O'Reilly. Cap. 9, secciones K-Means.

---
*Lectura relacionada con ML_U4_C01 · Secciones 2, 4 y 5*
*Lab ML_U4_Lab01 · Partes 1 y 2*
