# Ward, Distancias y la Matemática del Clustering Jerárquico

**Unidad 4 · Lectura complementaria 06 · Audiencia: doctorado**
versión: 2025-1 | modificado: 2026-05-30 | lectura estimada: 10 min

---

El criterio de Ward es el más utilizado en clustering jerárquico aglomerativo, y con razón: produce clusters compactos y de interpretación natural. Pero su derivación es raramente explicada en cursos introductorios. Esta lectura cierra esa brecha: formaliza Ward, lo conecta con K-means, examina sus propiedades teóricas y discute las alternativas para datos no euclidianos.

## El Criterio de Ward: Definición Formal

Ward propuso un criterio de fusión que responde a una pregunta simple: de todos los pares de clusters que podríamos fusionar, ¿cuál fusión produce el menor *daño* a la compacidad total de la partición?

**Definición:** El incremento de inercia al fusionar los clusters $C_i$ y $C_j$ es:
$$\Delta(C_i, C_j) = \text{WCSS}(C_i \cup C_j) - \text{WCSS}(C_i) - \text{WCSS}(C_j)$$

donde $\text{WCSS}(C) = \sum_{x \in C} \|x - \mu_C\|^2$ y $\mu_C$ es el centroide del cluster $C$.

Ward fusiona en cada paso el par $(C_i, C_j)$ que minimiza $\Delta(C_i, C_j)$.

**Forma cerrada de $\Delta$:** se puede demostrar algebraicamente que:
$$\Delta(C_i, C_j) = \frac{n_i n_j}{n_i + n_j} \|\mu_i - \mu_j\|^2$$

donde $n_i, n_j$ son los tamaños y $\mu_i, \mu_j$ los centroides de $C_i$ y $C_j$ respectivamente.

Esta expresión tiene una interpretación geométrica bella: $\Delta(C_i, C_j)$ es proporcional al cuadrado de la distancia entre centroides, *ponderado* por el producto de los tamaños de los clusters. Clusters grandes que están algo separados generan un $\Delta$ mayor que clusters pequeños igualmente separados. Ward "prefiere" fusionar clusters pequeños antes que grandes —lo que tiende a producir clusters de tamaño similar.

## Conexión con K-means

La conexión entre Ward y K-means es más profunda que la similitud superficial de "ambos minimizan WCSS".

**Teorema (Ward como K-means bottom-up):** La partición obtenida cortando el dendrograma de Ward a K clusters es un óptimo local del problema K-means. No es necesariamente el *global* —K-means++ con múltiples inicializaciones puede encontrar mejores soluciones— pero siempre es un óptimo local en el sentido de que ningún punto puede moverse a otro cluster para reducir el WCSS.

**Corolario:** Ward jerárquico con corte en K es una inicialización válida y frecuentemente buena para K-means. Tiene la ventaja de ser determinista (a diferencia de K-means++ aleatorio). En la práctica: Ward para inicializar + K-means para refinar es una estrategia robusta.

**Diferencia clave:** K-means con K dado busca el mínimo global del WCSS para ese K específico. Ward construye toda la jerarquía de particionamientos de forma greedy —en cada paso fusiona lo que conviene localmente, sin poder deshacer esa decisión. Esto hace que Ward sea subóptimo respecto a K-means puro, pero permite explorar todos los K simultáneamente.

## La Fórmula de Lance-Williams

El algoritmo aglomerativo naïve requiere recalcular todas las distancias entre clusters en cada fusión, lo que es $O(n^2)$ por paso y $O(n^3)$ total. La **fórmula de Lance-Williams (1967)** permite actualizar la distancia entre el nuevo cluster $(C_i \cup C_j)$ y cualquier otro cluster $C_k$ de manera recursiva, sin recalcular desde cero:

$$d(C_i \cup C_j, C_k) = \alpha_i \, d(C_i, C_k) + \alpha_j \, d(C_j, C_k) + \beta \, d(C_i, C_j) + \gamma \, |d(C_i, C_k) - d(C_j, C_k)|$$

Para Ward, los coeficientes son:
$$\alpha_i = \frac{n_i + n_k}{n_i + n_j + n_k}, \quad \alpha_j = \frac{n_j + n_k}{n_i + n_j + n_k}, \quad \beta = \frac{-n_k}{n_i + n_j + n_k}, \quad \gamma = 0$$

Esta recursión reduce la complejidad a $O(n^2 \log n)$, usando una estructura de heap para mantener las distancias mínimas actualizadas.

## La Propiedad Ultrametric

Un dendrograma válido debe satisfacer la **desigualdad ultrametric**: para cualesquiera tres clusters $A$, $B$, $C$:
$$h(A, B) \leq \max(h(A, C), h(B, C))$$

donde $h(A, B)$ es la altura a la que $A$ y $B$ se fusionan en el dendrograma.

Esta propiedad es más fuerte que la desigualdad triangular ordinaria ($d(A,B) \leq d(A,C) + d(C,B)$) —la ultrametric dice que la distancia entre $A$ y $B$ es como máximo el *máximo* de las distancias a través de $C$, no la suma.

**Implicación:** las distancias del dendrograma no son distancias euclidianas en el espacio original. Son las distancias cofenéticas, que forman una ultrametric en el espacio de la jerarquía. La distorsión entre las distancias originales y las cofenéticas es lo que mide la **correlación cofenética** —el coeficiente de Pearson entre los dos conjuntos de distancias.

**Cuándo importa:** si quieres que el dendrograma sea una representación fiel de las distancias originales, busca alta correlación cofenética (> 0.8). Si lo que te importa son los clusters finales, la correlación cofenética importa menos —Ward puede tener correlación moderada pero producir excelentes clusters.

## Clustering Jerárquico con Métricas no Euclidianas

Un punto de fuerza importante del clustering jerárquico respecto a K-means: puede funcionar con **cualquier matriz de distancias**, incluyendo métricas no euclidianas donde la noción de "centroide" no existe.

Algunos ejemplos importantes:

**Distancia de Levenshtein (edición):** entre cadenas de texto o secuencias de ADN. No existe un "promedio" de dos secuencias que sea una secuencia natural. Complete o average linkage funcionan directamente sobre la matriz de distancias. Ward no, porque requiere centroides.

**Distancia de árbol filogenético:** en biología evolutiva, las especies se agrupan por su historia evolutiva, no por features numéricas. La matriz de distancias puede construirse desde análisis de secuencias genómicas.

**Correlación como similitud:** en finanzas, los activos se agrupan por correlación de retornos. La "distancia" es $d_{ij} = \sqrt{2(1 - \rho_{ij})}$ donde $\rho$ es la correlación. Esta métrica satisface la desigualdad triangular pero no es euclidiana.

Para todas estas situaciones, K-means no aplica directamente pero el clustering jerárquico (con los criterios de enlace correctos) sí.

## Algoritmos Eficientes: SLINK y CLINK

Para datos grandes, los algoritmos de lance-williams son $O(n^2 \log n)$ en tiempo y $O(n^2)$ en espacio — el cuello de botella es almacenar la matriz de distancias completa. Existen algoritmos especializados que evitan esta limitación:

**SLINK** (Sibson, 1973): calcula single linkage en $O(n^2)$ tiempo y $O(n)$ espacio, sin construir la matriz completa. Explota la propiedad de que single linkage equivale al árbol de expansión mínima (MST — Minimum Spanning Tree) de la matriz de distancias.

**CLINK** (Defays, 1977): análogo para complete linkage, también $O(n^2)$ y $O(n)$.

No existe un algoritmo equivalentemente eficiente para Ward o average linkage —en general requieren $O(n^2)$ espacio. Para datasets muy grandes (> 100,000 puntos), se usa **Mini-batch agglomerative** o se aplica Ward en un subconjunto representativo.

## Para reflexionar

1. Demuestra algebraicamente que $\Delta(C_i, C_j) = \frac{n_i n_j}{n_i + n_j} \|\mu_i - \mu_j\|^2$. (Hint: expande $\text{WCSS}(C_i \cup C_j)$ usando la definición de varianza y la ley de la varianza total.)

2. Ward produce clusters de tamaño similar porque penaliza más la fusión de clusters grandes. ¿Puedes construir un contraejemplo donde Ward produce clusters de tamaños muy desiguales? ¿Qué condición sobre la geometría del dataset haría que Ward fusionara dos clusters grandes antes que uno grande con uno pequeño?

3. La correlación cofenética de single linkage es generalmente alta, pero single linkage falla frecuentemente en clustering de datos reales. ¿Qué conclusión extraes sobre el uso de la correlación cofenética como criterio para *seleccionar el método de enlace*? ¿Para qué sirve entonces la correlación cofenética?

## Para ir más lejos

- Ward, J. H. (1963). Hierarchical grouping to optimize an objective function. *Journal of the American Statistical Association*, 58(301), 236-244. [El paper original]

- Sibson, R. (1973). SLINK: An optimally efficient algorithm for the single-link cluster method. *The Computer Journal*, 16(1), 30-34.

- Lance, G. N., & Williams, W. T. (1967). A general theory of classificatory sorting strategies. *The Computer Journal*, 9(4), 373-380.

- Murtagh, F., & Legendre, P. (2014). Ward's hierarchical agglomerative clustering method: which algorithms implement Ward's criterion? *Journal of Classification*, 31(3), 274-295. [Clarifica confusiones en implementaciones]

- Müllner, D. (2013). fastcluster: Fast hierarchical, agglomerative clustering routines for R and Python. *Journal of Statistical Software*, 53(9). [Implementación eficiente disponible en Python]

---
*Lectura relacionada con ML_U4_C02 · Sección 3 (bloque 🟡) y Sección 2 (bloque 🟡)*
*Lab ML_U4_Lab02 · TODOs [PhD]*
