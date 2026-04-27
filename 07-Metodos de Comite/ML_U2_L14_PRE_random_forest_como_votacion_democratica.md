# Random Forest: Votación Democrática en Espacio de Características

**Unidad 2 · Lectura complementaria 14 · Audiencia: Pregrado**

versión: 2025-1 | modificado: 2026-04-24 | lectura estimada: 6 min

---

Imagina que quieres predecir si lloverá mañana. Consultas a cien meteorólogos profesionales, pero todos ellos tienen acceso exactamente a los mismos datos: temperaturas, presión barométrica, humedad. Todos usan la misma metodología. Sus predicciones serán idénticas. No has ganado sabiduría de multitudes; simplemente amplificaste un algoritmo. Ahora imagina un escenario distinto: cada uno de los cien meteorólogos recibe acceso únicamente a un subconjunto aleatorio de las variables disponibles. Uno ve solo temperatura y humedad; otro ve presión y velocidad del viento; otro ve solo la presión atmosférica de las últimas tres horas. Además, cada meteorólogo estudia únicamente un conjunto aleatorio de días históricos distintos. Con esta variabilidad forzada, cada experto llega a conclusiones diferentes, pero cuando promedias sus predicciones, obtienes algo que ninguno de ellos podría haber obtenido solo: una predicción robusta y confiable. Esto, en esencia, es Random Forest.

## El Poder Oculto de la Aleatoriedad Estructurada

Random Forest no es simplemente "entrenar muchos árboles de decisión". La diferencia sutil pero crítica es el mecanismo de diversidad. Si entrenaras cien árboles de decisión idénticos en el mismo dataset, obtendrías predicciones idénticas. El ensemble sería inútil. Random Forest, en cambio, introduce aleatoriedad deliberada en dos niveles.

Primero, cada árbol se entrena con una muestra bootstrap del dataset: un subconjunto aleatorio del mismo tamaño que el dataset original, muestreado con reemplazo. Esto significa que algunos datos aparecen múltiples veces en la muestra, y otros no aparecen nunca. Aunque parezca un desperdicio, esta variabilidad fuerza a cada árbol a aprender patrones ligeramente distintos. Segundo, en cada nodo del árbol donde decidimos hacer una división (split), Random Forest no considera todas las variables disponibles. En cambio, considera solo una cantidad aleatoria de variables: típicamente, la raíz cuadrada del número total de features. Si tienes diez características en tu dataset, cada nodo considera aleatoriamente solo tres de ellas como candidatos para la división.

Esta doble inyección de aleatoriedad tiene un propósito: garantizar que los árboles sean diferentes. Si todos los árboles vieran todas las variables y todos los datos, convergerían hacia el mismo modelo. Con la aleatoriedad, cada árbol dibuja su propia frontera de decisión, y el ensemble promedia estas fronteras.

## La Geometría de Fronteras Suavizadas

Un árbol de decisión individual dibuja fronteras ortogonales en el espacio de características. Si tu modelo tiene dos características —edad e ingresos— el árbol divide el espacio con líneas horizontales y verticales, creando rectángulos donde cada uno recibe una predicción. Estas fronteras son "dentadas" y rígidas, porque cada split divide una región en dos mitades mediante un corte perpendicular a un eje.

Imagina un dataset simple donde necesitas separar clientes que cancelarán su servicio de los que permanecerán. Un árbol individual produce una frontera con esquinas afiladas. Pero cuando entrenas cien árboles distintos, cada uno con acceso a variables y muestras diferentes, cada uno dibuja una frontera ligeramente distinta. El ensemble promedia estas cien fronteras, produciendo una frontera suavizada y mucho más realista. Las esquinas afiladas desaparecen, reemplazadas por curvas suaves que representan el verdadero patrón subyacente en los datos.

Esta suavización de fronteras es especialmente poderosa porque Random Forest no se compromete a una sola hipótesis geométrica. Los árboles individuales pueden cometer el error de sobreguiar sus límites de decisión (overfitting), pero el promedio del ensemble compensa estos errores idiosincráticos.

## El Valor Mágico: sqrt(d)

¿Por qué precisamente la raíz cuadrada del número de características? Esta pregunta tiene una respuesta práctica, aunque no completamente obvia. Si max_features es demasiado pequeño (por ejemplo, considerar solo una variable en cada split), los árboles son tan débiles y desconectados que necesitarías un número prohibitivamente grande de ellos para obtener buen desempeño. Si max_features es igual al número total de características, cada árbol considera todas las variables, y pierdes diversidad.

La raíz cuadrada es un equilibrio empírico que ha demostrado funcionar bien en la práctica. Intuitivamente, si diez de tus cincuenta variables son verdaderamente relevantes, al considerar aleatoriamente $\sqrt{50} \approx 7$ en cada split, cada árbol tiene buenas probabilidades de encontrar variables útiles, pero no tiene acceso garantizado a las mismas siete variables que el árbol anterior. Esto preserva diversidad sin sacrificar demasiado poder predictivo individual.

## Feature Importance: Leyendo la Mente del Ensemble

Un problema notorio con los ensambles es la interpretabilidad. Un árbol de decisión único es legible: puedes seguir una rama desde la raíz hasta una hoja y explicar exactamente qué regla el modelo usó para hacer su predicción. Con cien árboles, esta transparencia se evapora. ¿Cómo explicas por qué el modelo predijo un resultado si no puedes mostrar un único árbol coherente?

Random Forest ofrece una solución parcial: feature importance. Para cada variable, calcula cuánto cada split en cada árbol reduce la impureza (típicamente, medida por el índice de Gini). Luego promedia esta reducción ponderada por la frecuencia con que cada variable aparece como split en el ensemble. El resultado es una importancia numérica para cada variable: qué tan críales son en el conjunto de todos los árboles.

Esta métrica no es perfecta. A veces favorece a variables con muchas categorías o valores únicos. A veces correlaciona con variables realmente importantes en lugar de ser importante por sí misma. Pero es infinitamente mejor que no tener indicación alguna, y proporciona intuición sobre qué variables importan al modelo.

## Los Límites del Bosque

Random Forest es poderoso, pero no universal. Funciona excepcionalmente bien en datasets con muchas características numéricas y relaciones moderadamente complejas, especialmente cuando la cantidad de características es mayor que el número de muestras. Sin embargo, hay situaciones donde falla. Si tu espacio de características es muy pequeño (tres o cuatro variables), la aleatoriedad en la selección de variables produce poco beneficio. Si tus variables tienen relaciones fuertemente no lineales que dependen de interacciones complejas entre múltiples features simultáneamente, los árboles pueden no capturarlas: un árbol puede dividir por A y luego por B, pero no puede expresar fácilmente "A times B plus C elevado a la 2".

Además, Random Forest funciona mal cuando el dataset es minúsculo. El muestreo bootstrap amplifica el riesgo de sobreajuste si no tienes suficientes datos para que cada árbol vea variedad real. Y en problemas altamente desbalanceados (donde una clase es rara), el bootstrap puede inadvertidamente acentuar la clase mayoritaria en muestras de entrenamiento.

A pesar de estas limitaciones, Random Forest sigue siendo uno de los algoritmos más confiables en machine learning: simple, paralela, y frecuentemente una excelente línea base.

## Para reflexionar

1. Si tienes tres features y entrenas un Random Forest con max_features=1, cada árbol considera solo una variable por split. ¿Qué pasaría en el extremo opuesto, con max_features=3? ¿Cuál crees que tendría más varianza en las predicciones entre árboles?

2. Un Random Forest con n_estimators=500 requiere entrenar 500 árboles. ¿Crees que 500 es siempre mejor que 100? ¿Hay un punto de retornos decrecientes donde agregar más árboles casi no mejora el desempeño? ¿Cómo podrías identificarlo experimentalmente?

3. Imagina que debes explicar a un médico por qué tu modelo Random Forest predice "riesgo alto" de complicación quirúrgica en un paciente específico, sin poder mostrar una única secuencia lógica de decisiones como harías con un árbol. ¿Qué estrategias usarías para proporcionar una explicación creíble?

## Para ir más lejos

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. doi:10.1023/A:1010933404324
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly Media. Capítulo 7: Ensemble Learning and Random Forests.
- Louppe, G. (2014). *Understanding Random Forests: From Theory to Practice*. PhD Thesis, University of Liège. Disponible en arXiv:1407.7502.

---

*Lectura relacionada con la Clase ML_U2_C05 · Secciones 1-4 · Lab ML_U2_Lab04 · Assignment ML_A5*
