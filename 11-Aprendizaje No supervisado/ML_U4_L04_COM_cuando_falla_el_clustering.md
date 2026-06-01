# Cuando el Clustering Miente: Supuestos Rotos y Cómo Detectarlos

**Unidad 4 · Lectura complementaria 04 · Audiencia: todos**
versión: 2025-1 | modificado: 2026-05-30 | lectura estimada: 6 min

---

Hay una trampa silenciosa en los algoritmos de clustering: siempre encuentran grupos. Si le das K-means un dataset de números generados completamente al azar, te devolverá K clusters con tanta confianza como si los datos tuvieran estructura real. No hay alarma, no hay advertencia. El algoritmo hace su trabajo y el analista tiene que decidir si los grupos tienen sentido.

Esta trampa no es un defecto de implementación. Es una consecuencia fundamental de cómo funciona el clustering: es un proceso de optimización que minimiza una función objetivo, y esa función siempre tiene un mínimo aunque los datos sean basura.

## Los Supuestos que Nadie Menciona

Cada algoritmo de clustering lleva supuestos implícitos sobre la forma que deberían tener los grupos. Cuando esos supuestos se violan, el resultado es incorrecto —no de manera obvia, sino de manera sistemática y engañosa.

**K-means** asume que los clusters son:
- *Convexos*: sin agujeros, sin formas cóncavas
- *Isótropos* (esféricos): igualmente extendidos en todas las direcciones
- *De tamaño similar*: el número de puntos en cada cluster es comparable
- *Separados*: hay espacio vacío entre ellos

Cuando estas condiciones se cumplen, K-means funciona bien. Cuando se violan, K-means falla de maneras predecibles.

El caso más dramático es el de clusters en forma de luna o anillo. K-means no puede separar dos lunas entrelazadas porque sus fronteras de decisión son siempre hiperplanos. Sin importar cuántas iteraciones corra, la solución geométricamente incorrecta es un mínimo local de la inercia —y K-means lo encontrará.

El clustering jerárquico con Ward falla en los mismos casos que K-means, porque Ward minimiza el mismo WCSS. Single linkage, al seguir la cadena de puntos más cercanos, puede detectar clusters alargados y no convexos —pero a costa de ser frágil ante el encadenamiento (unir dos grupos grandes a través de un solo punto "puente").

## Tres Señales de Alerta

¿Cómo saber si el clustering que encontraste es genuino o un artefacto? No hay una prueba definitiva, pero hay señales que deben encender alarmas.

**Señal 1: el silhouette score es bajo para todo K razonable.** Si el silhouette máximo que puedes obtener es 0.2 o menos, los clusters son difusos. Los puntos están casi igual de cerca de su propio cluster que del vecino. En datos genuinamente agrupados, el silhouette del mejor K suele estar por encima de 0.4-0.5.

**Señal 2: el método del codo no muestra un codo.** Si la curva de inercia cae suavemente sin un quiebre claro, los datos probablemente no tienen estructura de clustering bien definida —o el número de clusters es ambiguo porque la separación entre grupos es gradual, no discreta.

**Señal 3: los clusters cambian dramáticamente con pequeñas perturbaciones.** Un clustering genuino debería ser relativamente estable: si añades un poco de ruido a los datos o cambias la semilla aleatoria, los mismos grupos deberían aparecer. Si los clusters cambian completamente con una inicialización diferente, son inestables y probablemente artefactos.

## La Maldición de la Alta Dimensión

Hay un problema adicional que se vuelve crítico en datos con muchas variables: en alta dimensión, las distancias euclidianas pierden su poder discriminativo.

El fenómeno se llama *concentración de la norma* o *maldición de la dimensionalidad*: en espacios de alta dimensión, las distancias entre todos los pares de puntos tienden a converger a un mismo valor. Si la distancia máxima es apenas un poco mayor que la mínima, la noción de "vecino cercano" pierde sentido.

Consecuencia práctica: K-means en datos de alta dimensión (cientos o miles de variables) puede producir clusters que no son estadísticamente significativos, incluso si el silhouette parece razonable. La solución estándar es reducir la dimensionalidad primero (PCA, por ejemplo) y luego aplicar clustering en el espacio reducido. Las próximas semanas cubrirán exactamente esas técnicas.

## Alternativas para Casos Difíciles

Conocer las limitaciones de K-means permite elegir mejor.

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) no asume forma esférica: define clusters como regiones densas separadas por regiones dispersas. Puede encontrar clusters de forma arbitraria y etiqueta automáticamente los outliers como ruido. Su debilidad: es sensible a sus propios parámetros (epsilon y min_samples) y falla con clusters de densidades muy distintas.

**Clustering espectral** usa los vectores propios de la matriz de similitudes para transformar los datos a un espacio donde K-means funciona bien, incluso en los casos de lunas y anillos. Es más costoso computacionalmente ($O(n^3)$ en la versión básica) y requiere elegir el kernel de similitud.

**GMM** (ya cubierto) generaliza K-means a clusters elípticos. No resuelve el problema de formas no convexas pero sí el de isotropía.

## La Pregunta que Importa: ¿Para Qué Son los Clusters?

Al final, la validez de un clustering no es solo una pregunta estadística —es una pregunta del dominio. Un clustering con silhouette de 0.3 puede ser perfectamente útil si los clusters identificados corresponden a categorías que tienen sentido para el problema y permiten tomar decisiones distintas en cada uno.

La segmentación de clientes de un banco puede producir clusters que se solapan bastante en el espacio de features pero que generan estrategias comerciales radicalmente distintas. El análisis de genes puede producir clusters que el biólogo reconoce inmediatamente como grupos funcionales coherentes, aunque las métricas internas no sean espectaculares.

El criterio final no es el silhouette: es si los clusters son *útiles* para el problema que se está resolviendo.

## Para reflexionar

1. Diseña un dataset sintético en 2D (puedes describirlo con palabras o en pseudocódigo) donde K-means con K=3 produciría clusters completamente incorrectos. ¿Cuál de las limitaciones (forma, tamaño, densidad) estás explotando? ¿Qué algoritmo funcionaría mejor?

2. El Gap Statistic propone comparar la inercia observada con la inercia esperada bajo una distribución uniforme de referencia. ¿Por qué una distribución uniforme es una buena referencia? ¿Qué significaría si tu dataset tiene una inercia *mayor* que la esperada bajo la distribución uniforme?

3. Tienes un dataset con 500 variables y 200 muestras. ¿Tiene sentido aplicar K-means directamente? ¿Qué harías primero? ¿Qué riesgos específicos introduce la alta dimensión además de la concentración de normas?

## Para ir más lejos

- Ester, M. et al. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD 1996*. [El paper original de DBSCAN]

- Ng, A. Y., Jordan, M. I., & Weiss, Y. (2002). On spectral clustering: Analysis and an algorithm. *NeurIPS 2002*. [Clustering espectral]

- Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters in a data set via the gap statistic. *JRSS-B*, 63(2), 411-423.

- Aggarwal, C. C., & Reddy, C. K. (Eds.) (2013). *Data Clustering: Algorithms and Applications*. CRC Press. [Referencia comprensiva]

---
*Lectura relacionada con ML_U4_C01 · Sección 4 (Limitaciones) y ML_U4_C02 · Sección 3*
*Assignment ML_A9 · Preguntas P3 y P4*
