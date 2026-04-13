# Métricas de Error en Modelos de Regresión

En esta guía revisamos formalmente las métricas principales para cuantificar la diferencia entre los valores observados ($y_i$) o reales, y los valores generados como predicción por nuestro modelo ($\hat{y}_i$).

### 1. Error Residual Individual (Residual Error)
El error individual o residual ($e_i$) es la diferencia puntual exacta matemática que existe entre el valor verdadero de de una muestra $i$ y el valor predecido por el modelo. Es la base de nuestras métricas.

$$ e_i = y_i - \hat{y}_i \tag{1}  $$

---

### 2. Error Total (Sum of Residuals)
Es la suma aritmética simple de todos los errores individuales. Como métrica de evaluación general no es fiable, ya que los errores negativos y los errores positivos tienden a cancelarse matemáticamente.

$$ E = \sum_{i=1}^{m} (y_i - \hat{y}_i) \tag{2}  $$

---

### 3. Error Promedio (ME: Mean Error)
Calcula el promedio de la sumatoria de errores brutos. Comparte el gravísimo problema de la cancelación de signo con la Ecuación (2). Su uso principal en investigación es exclusivamente diagnóstico para detectar **sesgo direccional** (si consistentemente nuestro modelo predice sistemáticamente valores muy por encima o muy por debajo de la realidad, su media no oscilará cerca de $0$).

$$ ME = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i) \tag{3}  $$

---

### 4. Error Absoluto Medio (MAE: Mean Absolute Error)
Resuelve tajantemente el problema de la cancelación de error de signos aplicando el valor absoluto a la resta, promediando la diferencia real. A diferencia de fórmulas cuadráticas, la penalidad crece de forma perfectamente lineal. **Es la métrica más robusta recomendada cuando existen valores atípicos (*outliers*) severos en nuestros datos**, porque no magnifica los errores lejanos de forma exponencial.

$$ MAE = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i| \tag{4}  $$

---

### 5. Error Cuadrático Medio (MSE: Mean Squared Error)
Al aplicar el cuadrado a los residuales en lugar del valor absoluto, logramos dos propiedades críticas en la disciplina del Machine Learning:
1. **Propiedad Analítica:** Genera una curva estrictamente convexa, continua y diferenciable, indispensable para poder ejecutar el cálculo de derivadas empleado en algoritmos de motor de optimización como el Gradiente Descendente.
2. **Propiedad de Penalización Magnificada:** Castiga y penaliza brutalmente a los *outliers*. Un error leve se mitiga ($0.1^2 = 0.01$), pero un error grosero dispara la penalización ($10^2 = 100$), forzando matemáticamente al modelo a ignorar ruidos pero predecir con alta convicción.

$$ MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 \tag{5}  $$

---

### 6. Raíz del Error Cuadrático Medio (RMSE: Root Mean Squared Error)
La métrica MSE altera y rompe dimensionalmente los datos (es decir, si tu objetivo $y$ está medido en *dólares*, el MSE devuelve una métrica irreal incomprensible medida en *dólares cuadrados*). 
Aplicando la raíz cuadrada al MSE forzamos el colapso dimensional de regreso a su estado normal. El RMSE permite que un humano logre leer, interpretar y explicar en las unidades originales el grado de equivocación promedio del algoritmo.

$$ RMSE = \sqrt{MSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2} \tag{6}  $$
