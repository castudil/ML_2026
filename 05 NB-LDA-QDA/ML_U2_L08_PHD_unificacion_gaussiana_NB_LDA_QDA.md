# NB, LDA y QDA como casos especiales de un mismo modelo gaussiano

**Unidad 2 · Lectura complementaria 8 · Audiencia: Doctorado**
versión: 2025-1 | modificado: 2026-04-06 | lectura estimada: 11 min

---

Una de las formas más poderosas de entender un conjunto de modelos no es estudiarlos por separado, sino encontrar el **modelo unificador** del que todos son casos especiales. Para NB, LDA y QDA, ese modelo unificador existe y tiene una forma elegante.

Esta lectura construye ese marco y extrae consecuencias que no son evidentes cuando los modelos se estudian por separado: cuántos parámetros realmente estima cada uno, qué restricciones imponen esos parámetros sobre la frontera de decisión, y cómo pensar en la regularización de QDA como un camino continuo desde QDA hasta LDA hasta NB.

---

## El modelo unificador: gaussiana multivariada por clase

Los tres modelos asumen que dentro de cada clase $k$, los datos siguen una distribución gaussiana multivariada:

$$P(X = x \mid Y = k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\!\left(-\frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)\right)$$

La diferencia entre los tres modelos está **exclusivamente en qué restricciones se imponen sobre** $\Sigma_k$.

| Modelo | Restricción sobre $\Sigma_k$ | Parámetros libres de covarianza |
|--------|------------------------------|--------------------------------|
| QDA | $\Sigma_k$ es cualquier PSD | $K \cdot \frac{d(d+1)}{2}$ |
| LDA | $\Sigma_k = \Sigma$ (igual para todos $k$) | $\frac{d(d+1)}{2}$ |
| GNB | $\Sigma_k = \text{diag}(\sigma_{k1}^2, \ldots, \sigma_{kd}^2)$ | $K \cdot d$ |

Cada restricción es una **hipótesis estadística** sobre la estructura de los datos. Si esa hipótesis es verdadera, el modelo se beneficia de la reducción de parámetros. Si es falsa, paga el costo del sesgo introducido.

---

## La función discriminante unificada

Para los tres modelos, la regla de decisión es:

$$\hat{y} = \arg\max_k \delta_k(x)$$

donde la función discriminante de clase $k$ es:

$$\delta_k(x) = -\frac{1}{2}\log|\Sigma_k| - \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) + \log \pi_k$$

Esta es la **función discriminante cuadrática general** (de QDA). Los otros modelos son restricciones sobre ella:

**Para LDA:** Como $\Sigma_k = \Sigma$ para todo $k$, el término $-\frac{1}{2}\log|\Sigma_k|$ es constante y cancela en la comparación. El término cuadrático $x^T \Sigma^{-1} x$ también es igual para todas las clases y cancela. Queda solo la parte **lineal** en $x$:

$$\delta_k^{LDA}(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2}\mu_k^T \Sigma^{-1} \mu_k + \log \pi_k$$

**Para GNB:** Como $\Sigma_k$ es diagonal, $\Sigma_k^{-1}$ también lo es y no hay términos cruzados $x_i x_j$ en la forma cuadrática. Esto significa que la frontera no puede ser una elipse inclinada — solo puede ser una elipse alineada con los ejes.

---

## Conteo de parámetros y su implicancia

El número de parámetros determina cuántos datos necesita cada modelo para estimarlos de forma estable. Una regla empírica útil es que se necesitan al menos 5–10 observaciones por parámetro libre para una estimación razonable.

Para un problema con $d = 10$ features y $K = 3$ clases:

- **GNB:** $K \cdot 2d = 3 \cdot 20 = 60$ parámetros (medias + varianzas por feature y clase)
- **LDA:** $\frac{d(d+1)}{2} + Kd = 55 + 30 = 85$ parámetros ($\Sigma$ compartida + $K$ vectores de media)
- **QDA:** $K \cdot \frac{d(d+1)}{2} + Kd = 165 + 30 = 195$ parámetros

Con 100 muestras de entrenamiento y 3 clases (~33 por clase), GNB necesita ~3 observaciones por parámetro de covarianza — manejable. QDA necesita ~0.5 observaciones por parámetro — el ajuste será severamente inestable. La matrix $\Sigma_k$ no solo será ruidosa: puede ser singular (no invertible).

Esta es la razón por la que sklearn implementa `QuadraticDiscriminantAnalysis(reg_param=α)`: añade $\alpha$ a la diagonal de $\Sigma_k$ antes de invertir, lo que garantiza que sea invertible y regulariza la estimación hacia la identidad.

---

## El camino continuo entre modelos: RDA

Existe un modelo que interpola de forma continua entre LDA y QDA: el **Regularized Discriminant Analysis** (RDA), propuesto por Friedman en 1989.

RDA usa una matriz de covarianza interpolada:

$$\hat{\Sigma}_k(\alpha) = (1 - \alpha)\hat{\Sigma}_k + \alpha \hat{\Sigma}$$

donde $\hat{\Sigma}_k$ es la covarianza estimada de la clase $k$, $\hat{\Sigma}$ es la covarianza pooled de LDA, y $\alpha \in [0,1]$ es un hiperparámetro.

Con $\alpha = 0$ se recupera QDA. Con $\alpha = 1$ se recupera LDA. Para valores intermedios, RDA es un **clasificador cuadrático regularizado** que se ajusta al nivel de datos disponibles.

Hay una segunda regularización posible: la interpolación con la identidad escalada,

$$\hat{\Sigma}_k(\alpha, \gamma) = (1-\gamma)\hat{\Sigma}_k(\alpha) + \gamma \frac{\text{tr}(\hat{\Sigma}_k(\alpha))}{d} I$$

que empuja las covarianzas hacia la identidad (features independientes con misma varianza). Con $\gamma = 1$ y $\alpha = 1$ se recupera algo cercano a GNB.

Este continuo es la motivación teórica para pensar en NB, LDA y QDA no como tres modelos discretos sino como puntos en un espacio de regularización.

---

## Estimación MLE de los parámetros

Para completar el cuadro, los estimadores de máxima verosimilitud de los parámetros comunes son:

$$\hat{\pi}_k = \frac{n_k}{n}, \qquad \hat{\mu}_k = \frac{1}{n_k}\sum_{i: y_i=k} x_i$$

La covarianza de QDA:
$$\hat{\Sigma}_k = \frac{1}{n_k} \sum_{i: y_i=k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$$

La covarianza pooled de LDA (estimador de $\Sigma$ compartida):
$$\hat{\Sigma} = \frac{1}{n - K} \sum_k \sum_{i: y_i=k} (x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$$

Nótese que LDA usa $n - K$ en el denominador (no $n$) para obtener un estimador insesgado de la covarianza dentro de cada clase.

---

## Conexión con análisis factorial y PCA

Hay una conexión profunda que vale la pena mencionar. LDA maximiza la razón de dispersión inter-clase a intra-clase, lo que equivale a resolver un problema de valores propios generalizados:

$$S_B w = \lambda S_W w$$

donde $S_B = \sum_k n_k (\hat{\mu}_k - \hat{\mu})(\hat{\mu}_k - \hat{\mu})^T$ es la dispersión entre clases y $S_W = \sum_k \sum_{i:y_i=k}(x_i - \hat{\mu}_k)(x_i - \hat{\mu}_k)^T$ es la dispersión dentro de clases.

Este problema produce los **ejes discriminantes** de LDA, que son distintos a los componentes de PCA. PCA maximiza la varianza total sin considerar las etiquetas. LDA maximiza la varianza entre clases relativa a la varianza dentro de clases — es una reducción de dimensionalidad **supervisada**.

---

## Para investigar

> 🔬 Friedman (1989) propone RDA como alternativa a QDA cuando $n/d$ es pequeño. Reproduce su experimento central: compara la tasa de error de LDA, QDA y RDA($\alpha^*$) en un dataset con $K=3$, $d=10$ y $n$ variando de 30 a 300 por clase. ¿En qué rango de $n$ RDA supera consistentemente a ambos extremos?

> 🔬 El estimador de Ledoit-Wolf ofrece una alternativa al shrinkage manual de RDA. Implementa `LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')` y compáralo con RDA con $\alpha$ elegido por CV en el dataset Wine. ¿Los dos enfoques convergen al mismo $\alpha$ óptimo?

---

## Para ir más lejos

- Friedman, J. H. (1989). *Regularized Discriminant Analysis*. Journal of the American Statistical Association 84(405). doi:10.2307/2289860
- Hastie, Tibshirani & Friedman (2009). *The Elements of Statistical Learning*. Springer. Cap. 4.3. doi:10.1007/978-0-387-84858-7
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Cap. 4.2.
- Ledoit, O. & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis 88(2). doi:10.1016/S0047-259X(03)00096-4

---

*Lectura relacionada con la Clase ML_U2_C03 · Secciones 2–4 y el experimento PhD de correlación · Lab ML_U2_Lab02 · Bonus Doctorado*
