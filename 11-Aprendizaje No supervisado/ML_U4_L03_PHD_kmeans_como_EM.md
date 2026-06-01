# K-means como Caso Límite de EM: La Unificación Formal

**Unidad 4 · Lectura complementaria 03 · Audiencia: doctorado**
versión: 2025-1 | modificado: 2026-05-30 | lectura estimada: 10 min

---

K-means es frecuentemente presentado como un algoritmo heurístico —dos pasos que se alternan hasta convergencia. Esta presentación, aunque pedagógicamente útil, oculta su fundamento probabilístico. K-means es en realidad el caso límite determinista de un proceso de inferencia probabilística sobre mezclas de distribuciones gaussianas.

## El Modelo de Mezcla Gaussiana

Un **Modelo de Mezcla Gaussiana (GMM)** supone que los datos fueron generados por el siguiente proceso:

1. Elegir un componente $k$ con probabilidad $\pi_k$ (las proporciones de mezcla, con $\sum_k \pi_k = 1$)
2. Generar un punto $x$ desde la gaussiana $\mathcal{N}(x; \mu_k, \Sigma_k)$

La densidad marginal del modelo es:
$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)$$

El objetivo es encontrar los parámetros $\{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$ que maximizan el log-likelihood sobre los datos observados:
$$\log \mathcal{L} = \sum_{i=1}^n \log \left( \sum_{k=1}^K \pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k) \right)$$

Este problema no tiene solución de forma cerrada porque el log de una suma no se factoriza. El algoritmo **EM** lo resuelve iterativamente.

## El Algoritmo EM para GMM

El algoritmo EM introduce variables latentes $z_{ik} \in \{0, 1\}$ que indican qué componente generó el punto $x_i$. Dado que $z$ no se observa, EM optimiza la log-verosimilitud esperada.

**Paso E (Expectation):** calcula la probabilidad posterior de que el componente $k$ haya generado $x_i$, dadas las estimaciones actuales de los parámetros:
$$r_{ik} = \frac{\pi_k \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i; \mu_j, \Sigma_j)}$$

Este valor $r_{ik}$ se llama **responsabilidad** del componente $k$ por el punto $x_i$. Es un número suave entre 0 y 1: cada componente tiene alguna responsabilidad por cada punto.

**Paso M (Maximization):** actualiza los parámetros maximizando la log-verosimilitud esperada respecto a las responsabilidades calculadas:
$$N_k = \sum_{i=1}^n r_{ik}, \quad \pi_k^{\text{nuevo}} = \frac{N_k}{n}$$
$$\mu_k^{\text{nuevo}} = \frac{1}{N_k} \sum_{i=1}^n r_{ik} x_i$$
$$\Sigma_k^{\text{nuevo}} = \frac{1}{N_k} \sum_{i=1}^n r_{ik} (x_i - \mu_k^{\text{nuevo}})(x_i - \mu_k^{\text{nuevo}})^\top$$

EM garantiza que el log-likelihood nunca decrece entre iteraciones (por la desigualdad de Jensen), convergiendo a un máximo local.

## La Derivación de K-means

Ahora restringimos el modelo GMM. Supón que todas las matrices de covarianza son isotrópicas e iguales: $\Sigma_k = \sigma^2 I$ para algún $\sigma^2$ común. La responsabilidad del componente $k$ por $x_i$ es:

$$r_{ik} = \frac{\pi_k \exp\left(-\frac{\|x_i - \mu_k\|^2}{2\sigma^2}\right)}{\sum_{j=1}^K \pi_j \exp\left(-\frac{\|x_i - \mu_j\|^2}{2\sigma^2}\right)}$$

Ahora tomamos el límite $\sigma^2 \to 0$. En este límite, el término correspondiente al centroide más cercano domina exponencialmente sobre todos los demás (los demás van a cero más rápido). Formalmente:

$$\lim_{\sigma^2 \to 0} r_{ik} = \begin{cases} 1 & \text{si } k = \arg\min_j \|x_i - \mu_j\|^2 \\ 0 & \text{en otro caso} \end{cases}$$

Las responsabilidades suaves se convierten en **asignaciones duras**: cada punto pertenece exactamente a un cluster. Esto es exactamente el Paso E de K-means.

Con asignaciones duras ($r_{ik} \in \{0,1\}$), el Paso M del GMM se simplifica:
$$\mu_k^{\text{nuevo}} = \frac{\sum_{i: z_i=k} x_i}{|C_k|}$$

que es la media del cluster. Esto es exactamente el Paso M de K-means.

**Conclusión:** K-means es EM sobre una mezcla de gaussianas isotrópicas con varianza común tendiendo a cero. El límite $\sigma^2 \to 0$ convierte las asignaciones suaves en duras y la maximización de log-likelihood en minimización de inercia.

## Consecuencias de la Derivación

Esta derivación no es solo un ejercicio formal. Revela por qué K-means tiene las propiedades que tiene.

**Supuesto de isotropía:** K-means supone covarianzas $\sigma^2 I$ — clusters esféricos de igual varianza. El GMM con covarianzas libres generaliza esto a clusters elípticos. Cuando los clusters reales son alargados o tienen densidades muy distintas, la restricción isotrópica impone una geometría equivocada.

**Sensibilidad a outliers:** la media (paso M) es el estadístico óptimo para gaussianas, pero es sensible a valores extremos. Un outlier puede arrastrar significativamente el centroide. El GMM con covarianzas libres es igualmente sensible. Alternativas robustas: k-medoids (usa la mediana multivariate), mezclas de distribuciones con colas pesadas (t de Student).

**La función objetivo que realmente se optimiza:** bajo el modelo GMM con $\sigma^2 \to 0$, minimizar la inercia es equivalente a maximizar el log-likelihood marginal del modelo. K-means no es un algoritmo heurístico —es inferencia MAP en un modelo probabilístico bien definido.

## GMM vs. K-means: Cuándo Usar Cada Uno

| Aspecto | K-means | GMM |
|---------|---------|-----|
| **Asignaciones** | Duras (0/1) | Suaves (probabilidades) |
| **Forma de clusters** | Solo esférica | Elíptica (cualquier covarianza) |
| **Incertidumbre** | No modela | Probabilidades posteriores $r_{ik}$ |
| **Velocidad** | $O(nKd \cdot T)$ | Más lento (inversión de matrices) |
| **Robustez** | Frágil a outliers | Igual de frágil |
| **Aplicación** | Clustering rápido, K conocido | Densidad, incertidumbre importa |

Una ventaja frecuentemente subestimada del GMM es que produce probabilidades: $r_{ik}$ dice "el punto $x_i$ tiene 80% de probabilidad de ser del cluster $k$". Para puntos en la frontera entre clusters, esto es información valiosa. K-means los asigna con confianza artificial.

## El Criterio BIC para Selección de K en GMM

El GMM tiene una ventaja adicional para selección de K: los criterios de información (BIC, AIC) son aplicables directamente porque tenemos un modelo probabilístico con log-likelihood bien definido.

Para un GMM con $K$ componentes en $d$ dimensiones, el número de parámetros libres es:
- $K-1$ proporciones de mezcla (están restringidas a sumar 1)
- $K \cdot d$ parámetros de media
- $K \cdot d(d+1)/2$ parámetros de covarianza (si es libre)

$$\text{BIC}(K) = -2 \log \hat{\mathcal{L}}(K) + p_K \log n$$

Se elige el $K$ que minimiza BIC, penalizando la complejidad del modelo. A diferencia del método del codo, BIC tiene una justificación estadística formal.

Nota: BIC no es directamente aplicable a K-means porque K-means no maximiza un log-likelihood (su función objetivo es la inercia, no una verosimilitud). El **Gap Statistic** de Tibshirani et al. es una alternativa para K-means.

## Para reflexionar

1. Si la varianza $\sigma^2$ en el GMM isotrópico controla el "grado de suavidad" de las asignaciones, ¿qué valor de $\sigma^2$ produce asignaciones perfectamente uniformes (cada punto pertenece a todos los clusters con probabilidad $1/K$)? ¿Qué modelo degenerado corresponde a este límite?

2. En el paso M del GMM, la media ponderada $\mu_k = \sum_i r_{ik} x_i / N_k$ puede calcularse eficientemente como una actualización online. ¿Cómo modificarías esto para implementar un EM online (sin procesar todos los datos en cada paso)?

3. GMM puede colapsar —un componente "absorbe" un solo punto y su varianza va a cero, haciendo el log-likelihood infinito. Este fenómeno se llama *singularidad*. ¿Cómo lo detectarías empíricamente durante el entrenamiento? ¿Cómo lo prevendrías?

## Para ir más lejos

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society B*, 39(1), 1-38. [El paper fundacional del EM]

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Cap. 9.2 (GMM) y 9.3 (K-means como EM).

- Xu, L., & Jordan, M. I. (1996). On convergence properties of the EM algorithm for Gaussian mixtures. *Neural Computation*, 8(1), 129-151.

- Neal, R. M., & Hinton, G. E. (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. *Learning in Graphical Models*, 355-368. [EM como ascenso coordinado en una función objetivo]

---
*Lectura relacionada con ML_U4_C01 · Sección 2 (bloque 🟡)*
*Lab ML_U4_Lab01 · Bonus amarillo (Soft K-means)*
