# Cómo Leer un Dendrograma: El Árbol que Cuenta la Historia de tus Datos

**Unidad 4 · Lectura complementaria 05 · Audiencia: pregrado**
versión: 2025-1 | modificado: 2026-05-30 | lectura estimada: 6 min

---

El dendrograma es uno de los gráficos más informativos en el análisis de datos —y también uno de los más intimidantes a primera vista. Parece un árbol al revés, con ramas que se unen en alturas variables, y puede representar cientos o miles de puntos. Pero una vez que entiendes su lógica, el dendrograma revela de un solo vistazo una información que K-means nunca puede dar: la estructura completa de similitudes a todos los niveles.

## Anatomía de un Dendrograma

El dendrograma tiene tres elementos fundamentales.

**Las hojas** (en la parte inferior) representan los datos individuales: cada punto del dataset es una hoja. Si el dataset tiene 150 puntos, el dendrograma tiene 150 hojas.

**Los nodos internos** representan fusiones de clusters. Cada vez que dos clusters se unen, aparece un nodo interno que los conecta. Si el dataset tiene 150 puntos, hay exactamente 149 fusiones y 149 nodos internos.

**La altura del nodo** es la distancia a la que ocurrió la fusión. Esta es la dimensión más importante del dendrograma. Un nodo bajo indica que los dos clusters que se fusionaron eran muy similares entre sí. Un nodo alto indica que la fusión ocurrió entre clusters que ya eran bastante diferentes.

## La Regla de Oro: Brechas y Cortes

El dendrograma permite elegir cualquier número de clusters simplemente "cortando" el árbol con una línea horizontal. El número de líneas verticales que cruza esa línea horizontal es el número de clusters resultantes.

La pregunta práctica es: ¿a qué altura cortar?

La respuesta es buscar la **brecha más grande** en el eje vertical. Si hay un salto grande entre dos alturas de fusión consecutivas, eso indica que los clusters que se estaban fusionando eran muy distintos entre sí —lo que sugiere que deberían mantenerse separados. La brecha más grande señala el corte más "natural".

Concretamente: si los últimos tres saltos del dendrograma tienen alturas 0.8, 0.9 y 3.2, hay una brecha grande entre 0.9 y 3.2. Cortar justo por debajo de la brecha (a altura ~2.0, por ejemplo) daría los clusters correspondientes a las fusiones a 0.8 y 0.9.

Esta heurística es equivalente a buscar el K donde el dendrograma está en su nivel más "estable": muchas fusiones pequeñas seguidas de un salto grande implica que los clusters que están por formarse son fundamentalmente distintos.

## El Ejemplo de Iris

El dataset Iris ilustra muy bien la lectura del dendrograma. Con el criterio de Ward, el dendrograma de Iris muestra típicamente dos tramos bien diferenciados:

Una primera zona de fusiones a alturas bajas (0-3 aproximadamente) donde los puntos individuales se unen y forman clusters coherentes. Aquí es donde se consolidan las tres especies.

Luego un salto grande a alturas de 4-6, donde los grupos de Versicolor y Virginica se fusionan entre sí. Este es el corte que separa "las dos especies similares" de "Setosa".

Finalmente un salto aún mayor (8-10) donde el grupo {Versicolor + Virginica} se une con Setosa. Este es el corte que separa "todas las iris" como un solo grupo.

Si cortas en la brecha principal (~altura 5), obtienes K=2: {Setosa} y {Versicolor + Virginica}. Si cortas un poco más abajo (~altura 3), obtienes K=3: las tres especies. Ambos son cortes válidos —responden preguntas distintas.

## Dendrogramas Truncados

Con muchos datos (cientos o miles de puntos), mostrar todas las hojas hace el dendrograma ilegible. La solución es **truncarlo**: mostrar solo las últimas P fusiones (las que ocurren a mayor altura). El parámetro `p` en scipy controla esto.

En un dendrograma truncado, algunas hojas representan múltiples puntos (normalmente el número se indica entre paréntesis). La altura de las barras sigue siendo la distancia de fusión, así que la interpretación visual es la misma.

Una forma complementaria de visualizar la estructura es el **histograma de distancias de fusión**: graficar las alturas de las últimas N fusiones en orden creciente. Las brechas aparecen como saltos en la gráfica, y son más fáciles de detectar que en el dendrograma propiamente dicho.

## El Heatmap Ordenado: Una Vista Complementaria

El dendrograma describe *cuándo* se unen los clusters. Una visualización complementaria muy útil es el **heatmap de distancias**, donde las filas y columnas son los puntos del dataset y el color de cada celda es la distancia entre esos dos puntos.

Ordenando las filas y columnas según el dendrograma (las hojas aparecen en el mismo orden que en el dendrograma), los clusters genuinos aparecen como bloques diagonales de color oscuro (puntos muy cercanos entre sí). Si los clusters son reales y bien definidos, los bloques serán visibles incluso sin etiquetar.

Esta visualización es especialmente útil en biología computacional, donde se usa para mostrar grupos de genes o muestras que se expresan de manera similar.

## Dendrograma vs. Codo: Cuándo Preferir Cada Uno

El método del codo y el silhouette de K-means requieren ejecutar el algoritmo múltiples veces y elegir un K. El dendrograma muestra la estructura completa en una sola ejecución.

El dendrograma es preferible cuando:
- No tienes hipótesis sobre el número de clusters y quieres explorar múltiples K simultáneamente
- Quieres visualizar la jerarquía (grupos y subgrupos anidados)
- El dataset tiene N moderado (< 5000 puntos; con más, el cálculo de la matriz de distancias $O(N^2)$ se vuelve costoso)

El método del codo/silhouette es preferible cuando:
- El dataset es grande (> 10,000 puntos)
- Ya tienes hipótesis sobre el rango de K y solo quieres confirmar
- La interpretabilidad del dendrograma es secundaria

## Para reflexionar

1. Un dendrograma de Ward sobre 50 muestras biológicas muestra dos brechas grandes: una a altura 4 (separa los grupos en K=2) y una a altura 2 (separa en K=4). ¿Cómo decides entre K=2 y K=4? ¿Qué información adicional del dominio biológico podrías usar?

2. El dendrograma de single linkage sobre datos con outliers a menudo muestra un fenómeno llamado "grafting": las hojas se unen de una en una a un cluster grande, produciendo un dendrograma muy asimétrico. ¿Por qué ocurre esto? ¿Qué forma tiene el dendrograma de Ward sobre los mismos datos?

3. Si el dendrograma no muestra ninguna brecha grande —todas las fusiones ocurren a alturas similares— ¿qué podrías concluir sobre la estructura del dataset? ¿Estarías dispuesto a forzar un K en ese caso?

## Para ir más lejos

- Murtagh, F., & Legendre, P. (2014). Ward's hierarchical agglomerative clustering method: which algorithms implement Ward's criterion? *Journal of Classification*, 31(3), 274-295.

- Kaufman, L., & Rousseeuw, P. J. (2009). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley. [Cobertura comprensiva, incluye interpretación de dendrogramas]

- Géron, A. (2022). *Hands-On Machine Learning* (3ª ed.). O'Reilly. Cap. 9.

- Scipy documentation: [scipy.cluster.hierarchy](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)

---
*Lectura relacionada con ML_U4_C02 · Sección 4 (Dendrogramas)*
*Lab ML_U4_Lab02 · Parte 1 y TODO 1*
