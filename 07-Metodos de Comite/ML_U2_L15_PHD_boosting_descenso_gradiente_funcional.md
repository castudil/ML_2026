# Gradient Boosting: Descenso de Gradiente en Espacio Funcional

**Unidad 2 · Lectura complementaria 15 · Audiencia: Doctorado**

versión: 2025-1 | modificado: 2026-04-24 | lectura estimada: 7 min

---

En 1999, Jerome Friedman publicó un artículo que reconceptualizó completamente la teoría de boosting. Hasta ese momento, AdaBoost era un algoritmo con reglas notoriamente ad hoc: ponderas las muestras de entrenamiento según su error, entrenas un clasificador débil, lo combinas con pesos logarítmicos, repites. Las justificaciones eran empíricas y el algoritmo parecía fruto de ingeniería ingeniosa pero sin principios unificadores. Friedman hizo una pregunta deceptivamente simple: ¿y si todo esto es simplemente descenso de gradiente? No descenso de gradiente en el espacio euclidiano de parámetros donde habitualmente lo practicamos, sino descenso de gradiente en el espacio infinito-dimensional de funciones. Esa pregunta abrió un mundo de unificación teórica donde AdaBoost, gradient boosting, XGBoost y una familia entera de algoritmos aparecen como variaciones de un mismo principio fundamental.

## Gradientes en Espacios Funcionales

En optimización convencional, minimizamos una función de pérdida L(y, ŷ) ajustando un vector de parámetros θ en ℝ^d. El gradiente es ∇_θ L, un vector en ℝ^d, y hacemos descenso de gradiente: θ ← θ - η ∇_θ L. El aprendizaje de máquinas histórico operaba así: ajustar pesos en una red neuronal, coeficientes en una regresión, hiperplanos en una SVM.

Gradient boosting replantea el problema. En lugar de parametrizar el modelo mediante un vector finito θ, parametrizamos mediante una función F: X → ℝ. El objetivo es encontrar la función F* que minimiza E[L(y, F(x))] sobre el espacio de todas las funciones posibles. Este es un problema de optimización infinito-dimensional. No podemos evaluarlo en ℝ^d porque F no es un vector.

Pero podemos hacer descenso de gradiente. En cada iteración, calculamos la dirección que apunta "cuesta abajo" en el espacio de funciones. Esta dirección se llama el gradiente funcional: -∇_F L = -∂L(y, F(x))/∂F(x). Evaluado puntualmente en cada muestra (x_i, y_i), el gradiente es un número: -∂L(y_i, F(x_i))/∂F(x_i).

Ahora viene el truco: no podemos añadir directamente este gradiente a F porque el gradiente es un número por muestra, no una función. En cambio, ajustamos una función base (típicamente, un árbol de decisión) h_m que aproxime el gradiente: el árbol aprende a predecir el gradiente (los "pseudoresiduals"). Luego, actualizamos F_m ← F_{m-1} + η h_m, donde η es un learning rate.

## Pseudoresiduals: El Lenguaje del Gradiente

El concepto central de gradient boosting es el pseudoresidual. Para una muestra (x_i, y_i) y una predicción actual F_{m-1}(x_i), el pseudoresidual es r_{m,i} = -∂L(y_i, F(x_i))/∂F(x_i) evaluado en F = F_{m-1}.

Este objeto unifica todo. Si usas pérdida L2 (regresión euclidiana), L = (y - F)², entonces ∂L/∂F = -2(y - F), y el pseudoresidual es r_{m,i} = y_i - F_{m-1}(x_i): son los residuos ordinarios. El nuevo árbol h_m aprende a predecir el error que cometió el modelo anterior.

Si usas pérdida L1 (más robusta a outliers), L = |y - F|, entonces ∂L/∂F = -sign(y - F), y el pseudoresidual es r_{m,i} = sign(y_i - F_{m-1}(x_i)): solo el signo del error. El nuevo árbol aprende si el error fue positivo o negativo, no su magnitud exacta, lo que lo hace robusto.

Para clasificación con log-loss (entropía cruzada binaria), L = -y log p - (1-y) log(1-p), donde p = σ(F) es la probabilidad predicha, el gradiente es ∂L/∂F = p - y. El pseudoresidual es r_{m,i} = y_i - p_{m-1}(x_i): la diferencia entre la etiqueta verdadera y la probabilidad predicha. El árbol aprende a corregir predicciones de probabilidad.

Este marco unifica algoritmos aparentemente desconectados bajo un paraguas común: especifica la pérdida, calcula el pseudoresidual, entrena un árbol, actualiza el modelo. Distinto de AdaBoost, no necesitas re-ponderar muestras ni definir pesos de combinación complejos. El algoritmo fluye naturalmente de los principios de optimización.

## AdaBoost como Caso Especial

La perspectiva funcional ilumina por qué AdaBoost funciona. AdaBoost es, en realidad, gradient boosting con pérdida exponencial: L(y, F) = e^{-yF}. Bajo esta pérdida, el gradiente es ∂L/∂F = -y e^{-yF}. Los "pesos de muestra" que AdaBoost asigna iterativamente, w_i^{(m)} ∝ e^{-y_i F_{m-1}(x_i)}, son exactamente proporcionales al gradiente de la pérdida exponencial.

Cuando entrenamos un árbol ponderado (donde muestras con mayor w_i^{(m)} tienen mayor importancia en la optimización), estamos efectivamente estimando la dirección del gradiente funcional. Los "pesos logarítmicos" de combinación que AdaBoost usa, α_m = (1/2) log((1 - err_m)/err_m), emergen naturalmente como el tamaño de paso óptimo en la dirección del gradiente. Todo lo que parecía arbitrario en AdaBoost —los pesos, los coeficientes— aparece como consecuencia inevitable de optimizar la pérdida exponencial mediante descenso de gradiente.

## XGBoost y la Aproximación de Segundo Orden

XGBoost, desarrollado por Chen y Guestrin (2016), da un paso más: aproxima la pérdida no solo con su gradiente, sino también con su derivada segunda (Hessiano). Usando una expansión de Taylor:

L(y, F_{m-1} + h) ≈ L(y, F_{m-1}) + ∇L · h + (1/2) ∇²L · h² + Ω(h)

donde Ω(h) = γT + (1/2)λ Σ_{j=1}^T w_j² es un término de regularización que penaliza la complejidad del árbol (T es el número de hojas, w_j son los pesos).

Para cada hoja j del árbol h, es posible resolver analíticamente qué peso óptimo w_j* minimiza esta pérdida aproximada. No necesitas grid search; tienes una solución en forma cerrada. Además, puedes comparar la mejora de pérdida de distintos árboles candidatos antes de entrenarlos completamente.

Esta aproximación de segundo orden tiene implicaciones prácticas profundas. XGBoost converge más rápidamente que gradient boosting estándar porque "ve" mejor la topología de la función de pérdida. Además, el término de regularización explícito permite control fino: aumentar γ penaliza árboles con muchas hojas (evita sobreajuste), aumentar λ penaliza pesos grandes de las hojas (suaviza predicciones).

## Implicaciones de Pérdida Personalizada

La perspectiva funcional explica por qué puedes usar pérdidas personalizadas. Si quieres que tu modelo sea robusto a outliers, usa Huber loss. Si quieres predicciones de cuantiles (intervalos de confianza), usa quantile loss. Si quieres optimizar una métrica de negocio específica (por ejemplo, maximizar ingresos sujeto a restricción de fraude), puedes derivar una pérdida personalizada que capture ese objetivo. Basta con proporcionar el gradiente y el Hessiano; el algoritmo hace el descenso.

Este es el punto donde la teoría de optimización clásica se encuentra con la pragmática del machine learning industrial. Ya no estás restringido a pérdidas cómodas matemáticamente; estás optimizando directamente lo que importa.

## Para reflexionar

1. En el artículo original de Friedman (2001), propone usar shrinkage: multiplicar cada árbol por un factor de learning rate η < 1 antes de agregarlo: F_m ← F_{m-1} + η h_m. ¿Por qué reduce esto el sobreajuste? ¿Qué analogía tiene con el learning rate en descenso de gradiente clásico en redes neuronales?

2. Para clasificación binaria con entropía cruzada, la probabilidad predicha es p = σ(F) = 1/(1 + e^{-F}). Si derivas ∂L/∂F donde L = -y log p - (1-y) log(1-p), ¿cuál es el pseudoresidual exacto en términos de p y y? ¿Cómo interpreta esto: qué aprende el árbol en cada iteración?

3. XGBoost regulariza con Ω(h) = γT + (1/2)λ Σ_j w_j². Si aumentas γ (penalización por número de hojas), ¿cómo cambia la forma de los árboles óptimos? ¿Y si aumentas λ (penalización por magnitud de pesos)? ¿Cuál es la diferencia conceptual entre estas dos formas de regularización?

## Para ir más lejos

- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232. doi:10.1214/aos/1013203451
- Schapire, R., & Freund, Y. (2012). *Boosting: Foundations and Algorithms*. MIT Press. ISBN: 978-0262017183.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (KDD 2016), 785–794. doi:10.1145/2939672.2939785
- Mason, L., Baxter, J., Bartlett, P., & Frean, M. (1999). Functional gradient techniques for combining hypotheses. *Advances in Neural Information Processing Systems* (NIPS 1999), 221–227.

---

*Lectura relacionada con la Clase ML_U2_C05 · Secciones 1-4 · Lab ML_U2_Lab04 · Assignment ML_A5*
