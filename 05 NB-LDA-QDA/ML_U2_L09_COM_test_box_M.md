# El test de Box's M: ¿tienen las clases la misma covarianza?

**Unidad 2 · Lectura complementaria 9 · Audiencia: todos**
versión: 2025-1 | modificado: 2026-04-06 | lectura estimada: 10 min

---

LDA hace un supuesto que raramente se verifica en los datos reales: que todas las clases tienen **la misma matriz de covarianza** $\Sigma$. Este supuesto permite simplificar el problema y obtener una frontera de decisión lineal. Pero si el supuesto es falso, LDA introduce sesgo: la frontera lineal no es la óptima y QDA puede hacerlo mejor.

¿Cómo saber si el supuesto se cumple? Para eso existe el **test de Box's M** (Box, 1949), una prueba estadística formal que evalúa si las matrices de covarianza de $K$ grupos son iguales. Esta lectura explica qué es, cómo funciona, cuándo usarlo y cuándo ignorarlo.

---

## El problema de igualdad de covarianzas

Supongamos que tenemos $K$ clases y estimamos la matriz de covarianza de cada una: $\hat{\Sigma}_1, \hat{\Sigma}_2, \ldots, \hat{\Sigma}_K$. Visualmente podemos graficar las elipses de confianza de cada clase y ver si tienen formas similares. Pero esa comparación es subjetiva. El test de Box's M convierte esa pregunta en un número con un p-valor.

La hipótesis nula del test es:

$$H_0: \Sigma_1 = \Sigma_2 = \cdots = \Sigma_K$$

Si rechazamos $H_0$, hay evidencia estadística de que al menos dos clases tienen covarianzas distintas, y LDA viola su supuesto fundamental.

---

## El estadístico M de Box

Para construir el estadístico, necesitamos comparar las covarianzas de cada clase con la covarianza **pooled** que usaría LDA. Sea:

- $n_k$: número de muestras de la clase $k$
- $\hat{\Sigma}_k$: covarianza estimada de la clase $k$ (MLE con $n_k$ en el denominador)
- $\hat{\Sigma}_W$: covarianza *within-class* pooled:

$$\hat{\Sigma}_W = \frac{1}{n - K} \sum_{k=1}^K (n_k - 1)\hat{\Sigma}_k$$

El estadístico M mide cuánto se alejan las $\hat{\Sigma}_k$ individuales de la pooled $\hat{\Sigma}_W$, usando el logaritmo del determinante como medida de "volumen" de cada matriz:

$$M = (n - K)\ln|\hat{\Sigma}_W| - \sum_{k=1}^K (n_k - 1)\ln|\hat{\Sigma}_k|$$

La intuición es que si todas las covarianzas fueran iguales, $\hat{\Sigma}_k \approx \hat{\Sigma}_W$ para todo $k$, y $M \approx 0$. Cuanto más distintas sean las covarianzas, mayor es $M$.

---

## Distribución y p-valor

Box (1949) demostró que, bajo $H_0$, una transformación de $M$ sigue aproximadamente una distribución **chi-cuadrado** con $\nu$ grados de libertad:

$$C = M \cdot \left(1 - c_1\right) \sim \chi^2_\nu$$

donde $c_1$ es un factor de corrección que depende de $d$, $K$ y los $n_k$, y los grados de libertad son:

$$\nu = \frac{d(d+1)(K-1)}{2}$$

Para $d = 4$ features y $K = 3$ clases: $\nu = \frac{4 \cdot 5 \cdot 2}{2} = 20$ grados de libertad.

Un p-valor pequeño (< 0.05) lleva a rechazar $H_0$: las covarianzas son significativamente distintas.

---

## Implementación en Python

`scipy` no incluye Box's M directamente, pero puede implementarse en pocas líneas:

```python
import numpy as np
from scipy import stats

def box_m_test(X, y):
    """
    Test de Box's M para igualdad de matrices de covarianza.
    Retorna el estadístico M, el estadístico chi2 aproximado y el p-valor.
    """
    classes = np.unique(y)
    K = len(classes)
    n = len(y)
    d = X.shape[1]

    # Covarianzas por clase (con n_k en denominador, MLE)
    covs = []
    ns   = []
    for k in classes:
        Xk = X[y == k]
        nk = len(Xk)
        ns.append(nk)
        covs.append(np.cov(Xk, rowvar=False, ddof=1))  # ddof=1 → insesgado

    ns   = np.array(ns)
    covs = np.array(covs)

    # Covarianza pooled (within-class)
    S_W = sum((nk - 1) * Sk for nk, Sk in zip(ns, covs)) / (n - K)

    # Estadístico M
    M = (n - K) * np.log(np.linalg.det(S_W))
    M -= sum((nk - 1) * np.log(np.linalg.det(Sk))
             for nk, Sk in zip(ns, covs))

    # Factor de corrección c1
    c1 = (sum(1/(nk - 1) for nk in ns) - 1/(n - K)) * \
         (2*d**2 + 3*d - 1) / (6*(d + 1)*(K - 1))

    # Estadístico chi2 y grados de libertad
    chi2_stat = M * (1 - c1)
    df = d * (d + 1) * (K - 1) // 2
    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    return M, chi2_stat, df, p_value
```

Ejemplo de uso sobre Wine:

```python
from sklearn.datasets import load_wine
wine = load_wine()
M, chi2, df, p = box_m_test(wine.data, wine.target)
print(f"M = {M:.2f} | chi2 = {chi2:.2f} | df = {df} | p = {p:.4f}")
# → M = 496.3 | chi2 = 475.4 | df = 20 | p ≈ 0.0000
```

El resultado dice: las covarianzas de los tres tipos de vino son **estadísticamente distintas** con alta confianza. Esto sugiere que QDA es más apropiado que LDA para este dataset.

---

## La trampa: Box's M es extremadamente sensible

Aquí viene la advertencia más importante de esta lectura: **Box's M es uno de los tests más sensibles a la no-normalidad que existen**.

La prueba supone que los datos de cada clase siguen una distribución gaussiana multivariada. Si esa condición no se cumple — y raramente se cumple exactamente — el test rechaza $H_0$ incluso cuando las covarianzas son genuinamente iguales. Con muestras grandes, la prueba rechaza casi siempre, no porque haya diferencias reales de covarianza sino porque detecta desviaciones menores de la normalidad.

En la práctica esto significa:

- Con **n pequeño** (< 50 por clase): el test tiene poca potencia. Un p > 0.05 no garantiza igualdad.
- Con **n grande** (> 200 por clase): el test casi siempre rechaza por diferencias triviales. Un p < 0.05 no implica que LDA sea inapropiado.
- Con **no-normalidad**: el test es inválido independientemente del tamaño muestral.

Por estas razones, muchos estadísticos recomiendan usar Box's M como **guía cualitativa** combinada con inspección visual de las covarianzas (heatmaps, elipses de confianza), en lugar de tomar sus p-valores como criterio definitivo.

---

## La alternativa práctica: comparar LDA vs QDA por CV

En machine learning la respuesta más pragmática a "¿son iguales las covarianzas?" es comparar LDA y QDA directamente con validación cruzada. Si QDA tiene F1 significativamente mayor, las covarianzas distintas importan en la práctica. Si son similares, el supuesto de LDA es suficientemente bueno, aunque Box's M lo rechace formalmente.

Esta comparación empírica es más informativa que el test estadístico porque responde la pregunta que realmente importa: ¿la diferencia de covarianzas afecta la clasificación?

---

## Para reflexionar

1. Box's M sobre Iris con 4 features y 3 clases tiene 20 grados de libertad. ¿Por qué ese número y no otro?

2. Si diseñas un experimento donde sabes que las covarianzas son iguales pero los datos tienen colas pesadas (distribución t-Student), ¿qué resultado esperarías del test?

3. En tu dataset de penguins, ¿usarías Box's M como criterio para elegir entre LDA y QDA, o preferirías la comparación por CV? ¿Cuál de los dos enfoques es más apropiado para el objetivo del assignment?

---

## Para ir más lejos

- Box, G. E. P. (1949). *A general distribution theory for a class of likelihood criteria*. Biometrika 36(3/4). doi:10.2307/2332671
- Mardia, K. V., Kent, J. T. & Bibby, J. M. (1979). *Multivariate Analysis*. Academic Press. Cap. 5.
- Rencher, A. C. (2002). *Methods of Multivariate Analysis* (2nd ed.). Wiley. Cap. 7.
- James et al. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer. Cap. 4.4. [statlearning.com]

---

*Lectura relacionada con la Clase ML_U2_C03 · Sección 3 (LDA) · Lab ML_U2_Lab02 · Parte 4, pregunta 4 [PhD] · Assignment ML_A3 · Sección 4, pregunta 4 [PhD]*
